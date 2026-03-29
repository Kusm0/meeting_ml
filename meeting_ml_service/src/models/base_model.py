"""
Base model class for all classification models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(
        self,
        task: str,
        model_type: str,
        num_classes: int,
        class_names: List[str],
    ):
        """
        Initialize base model.

        Args:
            task: Task name ("decision", "topic_type", "da")
            model_type: Model type ("tfidf", "bert")
            num_classes: Number of output classes
            class_names: List of class names
        """
        self.task = task
        self.model_type = model_type
        self.num_classes = num_classes
        self.class_names = class_names
        self.model = None
        self.is_trained = False

    @abstractmethod
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
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training history dictionary
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predicted class indices
        """
        pass

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features

        Returns:
            Class probabilities array
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model from disk.

        Args:
            path: Directory path to load model from
        """
        pass

    def get_prediction_with_confidence(
        self, X: Any
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
        """
        Get predictions with confidence scores and all probabilities.

        Args:
            X: Input features

        Returns:
            Tuple of (predictions, confidence scores, probability dicts)
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        # Get confidence (max probability)
        confidence = np.max(probabilities, axis=1)

        # Create probability dictionaries
        prob_dicts = []
        for probs in probabilities:
            prob_dict = {
                self.class_names[i]: float(probs[i])
                for i in range(len(self.class_names))
            }
            prob_dicts.append(prob_dict)

        return predictions, confidence, prob_dicts

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"task={self.task}, "
            f"model_type={self.model_type}, "
            f"num_classes={self.num_classes}, "
            f"is_trained={self.is_trained})"
        )

