"""
Training pipeline for all models.
"""

import gc
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type
import numpy as np
import torch
from loguru import logger

from src.config.settings import settings
from src.preprocessing.text_processor import TextProcessor
from src.training.data_loader import DataLoader
from src.models.base_model import BaseModel
from src.models.tfidf_models import (
    DecisionDetectorTFIDF,
    TopicTypeClassifierTFIDF,
    DAClassifierTFIDF,
)
from src.models.bert_models import (
    DecisionDetectorBERT,
    TopicTypeClassifierBERT,
    DAClassifierBERT,
)
from src.metrics.calculator import MetricsCalculator


# Model registry
MODEL_REGISTRY: Dict[str, Dict[str, Type[BaseModel]]] = {
    "tfidf": {
        "decision": DecisionDetectorTFIDF,
        "topic_type": TopicTypeClassifierTFIDF,
        "da": DAClassifierTFIDF,
    },
    "bert": {
        "decision": DecisionDetectorBERT,
        "topic_type": TopicTypeClassifierBERT,
        "da": DAClassifierBERT,
    },
}


class Trainer:
    """
    Training pipeline for TF-IDF and BERT models.
    """

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        text_processor: Optional[TextProcessor] = None,
    ):
        """
        Initialize trainer.

        Args:
            data_loader: DataLoader instance
            text_processor: TextProcessor instance
        """
        self.data_loader = data_loader or DataLoader()
        self.text_processor = text_processor or TextProcessor()
        self.metrics_calculator = MetricsCalculator()

        # Ensure data is loaded
        self.data_loader.load_data()
        self.data_loader.split_data()

    def train_model(
        self,
        model_type: str,
        task: str,
        **model_kwargs,
    ) -> Dict[str, Any]:
        """
        Train a single model.

        Args:
            model_type: "tfidf" or "bert"
            task: "decision", "topic_type", or "da"
            **model_kwargs: Additional model parameters

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_type} model for {task}...")

        # Get model class
        model_cls = MODEL_REGISTRY.get(model_type, {}).get(task)
        if model_cls is None:
            raise ValueError(
                f"Unknown model: {model_type}/{task}"
            )

        # Get data
        train_texts, train_labels = self.data_loader.get_task_data(task, "train")
        val_texts, val_labels = self.data_loader.get_task_data(task, "val")
        test_texts, test_labels = self.data_loader.get_task_data(task, "test")

        # Initialize model
        model = model_cls(**model_kwargs)

        # Prepare features
        if model_type == "tfidf":
            # Fit vectorizer on training data
            self.text_processor.fit_vectorizer(train_texts)

            # Transform
            X_train = self.text_processor.vectorize(train_texts)
            X_val = self.text_processor.vectorize(val_texts)
            X_test = self.text_processor.vectorize(test_texts)

            # Train
            history = model.train(X_train, train_labels, X_val, val_labels)

            # Evaluate
            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            test_preds = model.predict(X_test)

            train_probs = model.predict_proba(X_train)
            val_probs = model.predict_proba(X_val)
            test_probs = model.predict_proba(X_test)

        else:  # bert
            # Train with text directly
            history = model.train(
                train_texts, train_labels,
                val_texts, val_labels,
            )

            # Evaluate
            train_preds = model.predict(train_texts)
            val_preds = model.predict(val_texts)
            test_preds = model.predict(test_texts)

            train_probs = model.predict_proba(train_texts)
            val_probs = model.predict_proba(val_texts)
            test_probs = model.predict_proba(test_texts)

        # Calculate metrics
        task_config = settings.tasks[task]
        is_binary = task_config["type"] == "binary"

        train_metrics = self.metrics_calculator.calculate_all(
            train_labels, train_preds, train_probs,
            class_names=task_config["class_names"],
            is_binary=is_binary,
        )
        val_metrics = self.metrics_calculator.calculate_all(
            val_labels, val_preds, val_probs,
            class_names=task_config["class_names"],
            is_binary=is_binary,
        )
        test_metrics = self.metrics_calculator.calculate_all(
            test_labels, test_preds, test_probs,
            class_names=task_config["class_names"],
            is_binary=is_binary,
        )

        # Save model
        model_path = settings.models_dir / model_type / task
        model.save(model_path)

        # Save preprocessor for TF-IDF
        if model_type == "tfidf":
            preprocessor_path = settings.models_dir / "preprocessor.pkl"
            self.text_processor.save(preprocessor_path)

        # Prepare results
        results = {
            "model_type": model_type,
            "task": task,
            "history": history,
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
        }

        # Save metrics
        self._save_metrics(results, model_type, task)

        logger.info(
            f"Training complete for {model_type}/{task}. "
            f"Test accuracy: {test_metrics['accuracy']:.4f}"
        )

        # Clean up memory after BERT training
        if model_type == "bert":
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Memory cleaned up")

        return results

    def _save_metrics(
        self,
        results: Dict[str, Any],
        model_type: str,
        task: str,
    ) -> None:
        """Save training metrics to disk."""
        metrics_path = settings.metrics_dir / model_type / task

        # Save metrics JSON
        with open(metrics_path / "metrics.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable = self._make_serializable(results)
            json.dump(serializable, f, indent=2)

        # Save training history
        with open(metrics_path / "training_history.json", "w") as f:
            history = self._make_serializable(results.get("history", {}))
            json.dump(history, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        return obj

    def train_all_tfidf(self) -> Dict[str, Dict[str, Any]]:
        """Train all TF-IDF models."""
        results = {}

        for task in ["decision", "topic_type", "da"]:
            results[task] = self.train_model("tfidf", task)

        return results

    def train_all_bert(self) -> Dict[str, Dict[str, Any]]:
        """Train all BERT models."""
        results = {}

        for task in ["decision", "topic_type", "da"]:
            results[task] = self.train_model("bert", task)

        return results

    def train_all(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Train all models (TF-IDF and BERT)."""
        return {
            "tfidf": self.train_all_tfidf(),
            "bert": self.train_all_bert(),
        }


def get_model(model_type: str, task: str) -> BaseModel:
    """
    Get a trained model.

    Args:
        model_type: "tfidf" or "bert"
        task: "decision", "topic_type", or "da"

    Returns:
        Loaded model
    """
    model_cls = MODEL_REGISTRY.get(model_type, {}).get(task)
    if model_cls is None:
        raise ValueError(f"Unknown model: {model_type}/{task}")

    model = model_cls()
    model_path = settings.models_dir / model_type / task
    model.load(model_path)

    return model

