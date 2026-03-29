"""
Inference predictor module.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from src.config.settings import settings
from src.preprocessing.text_processor import TextProcessor
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


# Model registry
MODEL_CLASSES = {
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


class Predictor:
    """
    Predictor for all models.
    Handles loading models and making predictions.
    """

    def __init__(self):
        """Initialize predictor."""
        self.models: Dict[str, Dict[str, BaseModel]] = {
            "tfidf": {},
            "bert": {},
        }
        self.text_processor: Optional[TextProcessor] = None
        self._loaded_models: Dict[str, bool] = {}

    def load_preprocessor(self) -> None:
        """Load text preprocessor."""
        preprocessor_path = settings.models_dir / "preprocessor.pkl"

        if preprocessor_path.exists():
            self.text_processor = TextProcessor()
            self.text_processor.load(preprocessor_path)
            logger.info("Text processor loaded")
        else:
            logger.warning(
                f"Preprocessor not found at {preprocessor_path}. "
                "TF-IDF predictions will not work."
            )

    def load_model(self, model_type: str, task: str) -> bool:
        """
        Load a specific model.

        Args:
            model_type: "tfidf" or "bert"
            task: "decision", "topic_type", or "da"

        Returns:
            True if loaded successfully
        """
        key = f"{model_type}_{task}"

        if key in self._loaded_models:
            return self._loaded_models[key]

        model_path = settings.models_dir / model_type / task

        if not (model_path / "model.pkl").exists() and \
           not (model_path / "model.pt").exists():
            logger.warning(f"Model not found: {model_path}")
            self._loaded_models[key] = False
            return False

        try:
            model_cls = MODEL_CLASSES[model_type][task]
            model = model_cls()
            model.load(model_path)

            self.models[model_type][task] = model
            self._loaded_models[key] = True

            logger.info(f"Model loaded: {model_type}/{task}")
            return True

        except Exception as e:
            logger.error(f"Error loading model {model_type}/{task}: {e}")
            self._loaded_models[key] = False
            return False

    def load_all_models(self) -> Dict[str, bool]:
        """
        Load all available models.

        Returns:
            Dictionary mapping model key to load status
        """
        self.load_preprocessor()

        status = {}

        for model_type in ["tfidf", "bert"]:
            for task in ["decision", "topic_type", "da"]:
                key = f"{model_type}_{task}"
                status[key] = self.load_model(model_type, task)

        return status

    def predict_single(
        self,
        text: str,
        model_type: str,
        task: str,
    ) -> Dict[str, Any]:
        """
        Make prediction with a single model.

        Args:
            text: Input transcript
            model_type: "tfidf" or "bert"
            task: "decision", "topic_type", or "da"

        Returns:
            Prediction result dictionary
        """
        key = f"{model_type}_{task}"

        if key not in self._loaded_models or not self._loaded_models[key]:
            self.load_model(model_type, task)

        if not self._loaded_models.get(key, False):
            return {
                "error": f"Model {model_type}/{task} not available",
                "prediction": None,
                "confidence": None,
            }

        model = self.models[model_type][task]
        task_config = settings.tasks[task]

        try:
            if model_type == "tfidf":
                if self.text_processor is None:
                    self.load_preprocessor()

                if self.text_processor is None:
                    return {
                        "error": "Text processor not available",
                        "prediction": None,
                        "confidence": None,
                    }

                X = self.text_processor.vectorize([text])
                preds = model.predict(X)
                probs = model.predict_proba(X)

            else:  # bert
                preds = model.predict([text])
                probs = model.predict_proba([text])

            pred_idx = int(preds[0])
            pred_name = task_config["class_names"][pred_idx]
            confidence = float(np.max(probs[0]))

            # Build probability dictionary
            probabilities = {
                task_config["class_names"][i]: float(probs[0][i])
                for i in range(len(task_config["class_names"]))
                if i < len(probs[0])
            }

            return {
                "prediction": pred_name,
                "prediction_idx": pred_idx,
                "confidence": confidence,
                "probabilities": probabilities,
                "model_type": model_type,
                "task": task,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "prediction": None,
                "confidence": None,
            }

    def predict_all_tasks(
        self,
        text: str,
        model_type: str = "tfidf",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Make predictions for all tasks with one model type.

        Args:
            text: Input transcript
            model_type: "tfidf" or "bert"

        Returns:
            Dictionary with predictions for each task
        """
        results = {}

        for task in ["decision", "topic_type", "da"]:
            results[task] = self.predict_single(text, model_type, task)

        return results

    def predict_all_models(
        self,
        text: str,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Make predictions with all models (TF-IDF and BERT).

        Args:
            text: Input transcript

        Returns:
            Dictionary with predictions for each model type and task
        """
        return {
            "tfidf": self.predict_all_tasks(text, "tfidf"),
            "bert": self.predict_all_tasks(text, "bert"),
        }

    def get_model_status(self) -> Dict[str, Dict[str, bool]]:
        """
        Get status of all models.

        Returns:
            Dictionary indicating which models are loaded
        """
        status = {
            "tfidf": {},
            "bert": {},
        }

        for model_type in ["tfidf", "bert"]:
            for task in ["decision", "topic_type", "da"]:
                key = f"{model_type}_{task}"
                status[model_type][task] = self._loaded_models.get(key, False)

        return status


# Global predictor instance
_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    """Get or create global predictor instance."""
    global _predictor

    if _predictor is None:
        _predictor = Predictor()
        _predictor.load_all_models()

    return _predictor

