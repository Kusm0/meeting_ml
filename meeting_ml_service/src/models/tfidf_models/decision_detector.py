"""
TF-IDF based Decision Detection model.
Binary classification: does the segment contain a decision?
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from loguru import logger

from src.models.base_model import BaseModel
from src.config.settings import settings


class DecisionDetectorTFIDF(BaseModel):
    """
    TF-IDF + Logistic Regression for Decision Detection.
    Binary classification task.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        C: float = 1.0,
        use_svm: bool = False,
    ):
        """
        Initialize Decision Detector.

        Args:
            max_iter: Maximum iterations for solver
            class_weight: Class weight strategy ("balanced" for imbalanced data)
            C: Regularization strength
            use_svm: Whether to use SVM instead of Logistic Regression
        """
        task_config = settings.tasks["decision"]
        super().__init__(
            task="decision",
            model_type="tfidf",
            num_classes=task_config["num_classes"],
            class_names=task_config["class_names"],
        )

        self.max_iter = max_iter
        self.class_weight = class_weight
        self.C = C
        self.use_svm = use_svm

        self._init_model()

    def _init_model(self) -> None:
        """Initialize the classifier."""
        if self.use_svm:
            # SVM with calibration for probability estimates
            base_svm = LinearSVC(
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                C=self.C,
                random_state=42,
            )
            self.model = CalibratedClassifierCV(base_svm, cv=3)
        else:
            self.model = LogisticRegression(
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                C=self.C,
                random_state=42,
                solver="lbfgs",
            )

    def train(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Optional[Any] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training TF-IDF features (sparse matrix)
            y_train: Training labels
            X_val: Validation features (optional, for logging)
            y_val: Validation labels (optional)

        Returns:
            Training history dictionary
        """
        logger.info(
            f"Training Decision Detector (TF-IDF) on "
            f"{X_train.shape[0]} samples..."
        )

        # Train
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Calculate training metrics
        train_preds = self.model.predict(X_train)
        train_acc = np.mean(train_preds == y_train)

        history = {
            "train_accuracy": float(train_acc),
            "train_samples": X_train.shape[0],
        }

        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_preds = self.model.predict(X_val)
            val_acc = np.mean(val_preds == y_val)
            history["val_accuracy"] = float(val_acc)
            history["val_samples"] = X_val.shape[0]

        logger.info(
            f"Training complete. Train accuracy: {train_acc:.4f}"
            + (f", Val accuracy: {history.get('val_accuracy', 'N/A')}"
               if X_val is not None else "")
        )

        return history

    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: TF-IDF features

        Returns:
            Predicted class indices (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: TF-IDF features

        Returns:
            Class probabilities [P(no_decision), P(has_decision)]
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        return self.model.predict_proba(X)

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / "model.pkl"
        config_path = path / "config.pkl"

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        # Save config
        config = {
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "C": self.C,
            "use_svm": self.use_svm,
            "is_trained": self.is_trained,
        }
        with open(config_path, "wb") as f:
            pickle.dump(config, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Directory path to load model from
        """
        path = Path(path)
        model_path = path / "model.pkl"
        config_path = path / "config.pkl"

        # Load model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load config
        with open(config_path, "rb") as f:
            config = pickle.load(f)

        self.max_iter = config["max_iter"]
        self.class_weight = config["class_weight"]
        self.C = config["C"]
        self.use_svm = config["use_svm"]
        self.is_trained = config["is_trained"]

        logger.info(f"Model loaded from {path}")

    def get_feature_importance(
        self, feature_names: List[str], top_k: int = 20
    ) -> Dict[str, List[tuple]]:
        """
        Get feature importance for interpretation.

        Args:
            feature_names: List of feature names
            top_k: Number of top features to return

        Returns:
            Dictionary with positive/negative important features
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        if self.use_svm:
            # Get coefficients from calibrated classifier
            coef = self.model.calibrated_classifiers_[0].estimator.coef_[0]
        else:
            coef = self.model.coef_[0]

        # Get top positive features (indicate decision)
        top_positive_idx = np.argsort(coef)[-top_k:][::-1]
        top_positive = [
            (feature_names[i], float(coef[i]))
            for i in top_positive_idx
        ]

        # Get top negative features (indicate no decision)
        top_negative_idx = np.argsort(coef)[:top_k]
        top_negative = [
            (feature_names[i], float(coef[i]))
            for i in top_negative_idx
        ]

        return {
            "positive": top_positive,
            "negative": top_negative,
        }

