"""
Metrics visualization module.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from src.config.settings import settings


class MetricsVisualizer:
    """
    Visualize training metrics and model performance.
    """

    def __init__(self, figsize: tuple = (10, 6)):
        self.figsize = figsize
        plt.style.use("seaborn-v0_8-whitegrid")

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot confusion matrix heatmap.

        Args:
            cm: Confusion matrix array
            class_names: List of class names
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")

        return fig

    def plot_learning_curves(
        self,
        history: Dict[str, List[float]],
        title: str = "Learning Curves",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot training and validation learning curves.

        Args:
            history: Training history with loss and accuracy
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(history.get("train_loss", [])) + 1)

        # Loss plot
        if "train_loss" in history:
            axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
        if "val_loss" in history:
            axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        if "train_accuracy" in history:
            axes[1].plot(
                epochs, history["train_accuracy"], "b-", label="Train Accuracy"
            )
        if "val_accuracy" in history:
            axes[1].plot(
                epochs, history["val_accuracy"], "r-", label="Val Accuracy"
            )

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curves")
        axes[1].legend()
        axes[1].grid(True)

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Learning curves saved to {save_path}")

        return fig

    def plot_roc_curve(
        self,
        fpr: List[float],
        tpr: List[float],
        roc_auc: float,
        title: str = "ROC Curve",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot ROC curve.

        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under ROC curve
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(fpr, tpr, "b-", label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "r--", label="Random")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curve saved to {save_path}")

        return fig

    def plot_pr_curve(
        self,
        precision: List[float],
        recall: List[float],
        pr_auc: float,
        title: str = "Precision-Recall Curve",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.

        Args:
            precision: Precision values
            recall: Recall values
            pr_auc: Area under PR curve
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(recall, precision, "b-", label=f"PR (AUC = {pr_auc:.3f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"PR curve saved to {save_path}")

        return fig

    def plot_class_distribution(
        self,
        distribution: Dict[str, int],
        title: str = "Class Distribution",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot class distribution bar chart.

        Args:
            distribution: Dictionary mapping class name to count
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        classes = list(distribution.keys())
        counts = list(distribution.values())

        bars = ax.bar(classes, counts, color="steelblue")

        # Add value labels
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.set_title(title)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Class distribution saved to {save_path}")

        return fig

    def plot_metrics_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_names: List[str] = ["accuracy", "precision", "recall", "f1"],
        title: str = "Model Comparison",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot metrics comparison across models.

        Args:
            metrics: Dict mapping model name to metrics dict
            metric_names: List of metrics to compare
            title: Plot title
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(metric_names))
        width = 0.8 / len(metrics)

        for i, (model_name, model_metrics) in enumerate(metrics.items()):
            values = [model_metrics.get(m, 0) for m in metric_names]
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model_name)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, axis="y")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Metrics comparison saved to {save_path}")

        return fig

    def generate_all_plots(
        self,
        model_type: str,
        task: str,
        metrics: Dict[str, Any],
        history: Dict[str, List[float]],
        class_names: List[str],
    ) -> Dict[str, plt.Figure]:
        """
        Generate all plots for a model.

        Args:
            model_type: "tfidf" or "bert"
            task: Task name
            metrics: Metrics dictionary
            history: Training history
            class_names: List of class names

        Returns:
            Dictionary of figures
        """
        plots_dir = settings.metrics_dir / model_type / task / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        figures = {}

        # Confusion matrix
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            fig = self.plot_confusion_matrix(
                cm, class_names,
                title=f"Confusion Matrix - {model_type.upper()} {task}",
                save_path=plots_dir / "confusion_matrix.png",
            )
            figures["confusion_matrix"] = fig

        # Learning curves (for BERT)
        if history and "train_loss" in history:
            fig = self.plot_learning_curves(
                history,
                title=f"Learning Curves - {model_type.upper()} {task}",
                save_path=plots_dir / "learning_curves.png",
            )
            figures["learning_curves"] = fig

        # ROC curve (for binary)
        if "roc_curve" in metrics:
            fig = self.plot_roc_curve(
                metrics["roc_curve"]["fpr"],
                metrics["roc_curve"]["tpr"],
                metrics.get("roc_auc", 0),
                title=f"ROC Curve - {model_type.upper()} {task}",
                save_path=plots_dir / "roc_curve.png",
            )
            figures["roc_curve"] = fig

        # PR curve (for binary)
        if "pr_curve" in metrics:
            fig = self.plot_pr_curve(
                metrics["pr_curve"]["precision"],
                metrics["pr_curve"]["recall"],
                metrics.get("pr_auc", 0),
                title=f"PR Curve - {model_type.upper()} {task}",
                save_path=plots_dir / "pr_curve.png",
            )
            figures["pr_curve"] = fig

        plt.close("all")

        return figures

    def load_and_visualize(
        self,
        model_type: str,
        task: str,
    ) -> Dict[str, plt.Figure]:
        """
        Load saved metrics and generate visualizations.

        Args:
            model_type: "tfidf" or "bert"
            task: Task name

        Returns:
            Dictionary of figures
        """
        metrics_path = settings.metrics_dir / model_type / task

        # Load metrics
        metrics_file = metrics_path / "metrics.json"
        if not metrics_file.exists():
            logger.warning(f"Metrics file not found: {metrics_file}")
            return {}

        with open(metrics_file, "r") as f:
            data = json.load(f)

        # Load history
        history_file = metrics_path / "training_history.json"
        history = {}
        if history_file.exists():
            with open(history_file, "r") as f:
                history = json.load(f)

        # Get class names
        task_config = settings.tasks.get(task, {})
        class_names = task_config.get("class_names", [])

        # Get test metrics
        test_metrics = data.get("metrics", {}).get("test", {})

        return self.generate_all_plots(
            model_type, task, test_metrics, history, class_names
        )

