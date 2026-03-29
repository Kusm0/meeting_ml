"""
Metrics calculation module.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from loguru import logger


class MetricsCalculator:
    """
    Calculate various classification metrics.
    """

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        is_binary: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate all metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            class_names: List of class names
            is_binary: Whether this is binary classification

        Returns:
            Dictionary with all metrics
        """
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

        # Precision, Recall, F1
        # For binary classification, use weighted average to handle class imbalance
        # This gives more informative metrics when the model struggles with minority class
        average = "weighted" if is_binary else "macro"

        metrics["precision"] = float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(y_true, y_pred, average=average, zero_division=0)
        )

        # F1 micro (for multiclass)
        if not is_binary:
            metrics["f1_micro"] = float(
                f1_score(y_true, y_pred, average="micro", zero_division=0)
            )
            metrics["f1_weighted"] = float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class metrics
        if class_names is not None:
            # Get unique labels in data
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            # Filter class names to only include present classes
            present_class_names = [
                class_names[i] for i in unique_labels 
                if i < len(class_names)
            ]
            
            try:
                report = classification_report(
                    y_true, y_pred,
                    labels=unique_labels,
                    target_names=present_class_names,
                    output_dict=True,
                    zero_division=0,
                )
                metrics["per_class"] = {
                    name: {
                        "precision": report[name]["precision"],
                        "recall": report[name]["recall"],
                        "f1": report[name]["f1-score"],
                        "support": report[name]["support"],
                    }
                    for name in present_class_names
                    if name in report
                }
            except Exception as e:
                logger.warning(f"Could not generate per-class metrics: {e}")

        # ROC-AUC and PR-AUC for binary classification
        if is_binary and y_proba is not None:
            try:
                # Get positive class probability
                if len(y_proba.shape) == 2:
                    pos_proba = y_proba[:, 1]
                else:
                    pos_proba = y_proba

                metrics["roc_auc"] = float(roc_auc_score(y_true, pos_proba))
                metrics["pr_auc"] = float(
                    average_precision_score(y_true, pos_proba)
                )

                # ROC curve points
                fpr, tpr, thresholds = roc_curve(y_true, pos_proba)
                metrics["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": thresholds.tolist(),
                }

                # Precision-Recall curve points
                precision, recall, thresholds = precision_recall_curve(
                    y_true, pos_proba
                )
                metrics["pr_curve"] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "thresholds": thresholds.tolist(),
                }
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")

        # Multiclass ROC-AUC
        if not is_binary and y_proba is not None:
            try:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(
                        y_true, y_proba,
                        multi_class="ovr",
                        average="macro",
                    )
                )
            except Exception as e:
                logger.warning(f"Could not calculate multiclass AUC: {e}")

        return metrics

    def calculate_class_distribution(
        self,
        y: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Calculate class distribution.

        Args:
            y: Labels
            class_names: Optional class names

        Returns:
            Dictionary mapping class to count
        """
        unique, counts = np.unique(y, return_counts=True)

        if class_names is not None:
            return {
                class_names[int(idx)]: int(count)
                for idx, count in zip(unique, counts)
                if int(idx) < len(class_names)
            }
        else:
            return {
                str(idx): int(count)
                for idx, count in zip(unique, counts)
            }

