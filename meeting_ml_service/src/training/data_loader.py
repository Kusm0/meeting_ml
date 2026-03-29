"""
Data loading and splitting module.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger

from src.config.settings import settings


class DataLoader:
    """
    Loads and prepares data for training.
    Handles stratified splitting by meeting_id to avoid data leakage.
    """

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42,
    ):
        """
        Initialize data loader.

        Args:
            dataset_path: Path to CSV dataset
            train_split: Training set proportion
            val_split: Validation set proportion
            test_split: Test set proportion
            random_state: Random seed for reproducibility
        """
        self.dataset_path = dataset_path or settings.dataset_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state

        self.df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from CSV.

        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading dataset from {self.dataset_path}...")

        self.df = pd.read_csv(self.dataset_path)

        logger.info(
            f"Loaded {len(self.df)} records from "
            f"{self.df['meeting_id'].nunique()} meetings"
        )

        return self.df

    def _get_meeting_split(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Split meetings into train/val/test sets.
        This ensures all segments from one meeting stay in the same split.

        Returns:
            Tuple of (train_meetings, val_meetings, test_meetings)
        """
        meetings = self.df["meeting_id"].unique()
        n_meetings = len(meetings)

        # First split: train vs (val + test)
        val_test_size = self.val_split + self.test_split
        train_meetings, val_test_meetings = train_test_split(
            meetings,
            test_size=val_test_size,
            random_state=self.random_state,
        )

        # Second split: val vs test
        relative_test_size = self.test_split / val_test_size
        val_meetings, test_meetings = train_test_split(
            val_test_meetings,
            test_size=relative_test_size,
            random_state=self.random_state,
        )

        logger.info(
            f"Meeting split: train={len(train_meetings)}, "
            f"val={len(val_meetings)}, test={len(test_meetings)}"
        )

        return (
            train_meetings.tolist(),
            val_meetings.tolist(),
            test_meetings.tolist(),
        )

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets by meeting_id.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            self.load_data()

        train_meetings, val_meetings, test_meetings = self._get_meeting_split()

        self.train_df = self.df[
            self.df["meeting_id"].isin(train_meetings)
        ].copy()
        self.val_df = self.df[
            self.df["meeting_id"].isin(val_meetings)
        ].copy()
        self.test_df = self.df[
            self.df["meeting_id"].isin(test_meetings)
        ].copy()

        logger.info(
            f"Data split: train={len(self.train_df)}, "
            f"val={len(self.val_df)}, test={len(self.test_df)}"
        )

        return self.train_df, self.val_df, self.test_df

    def get_task_data(
        self, task: str, split: str = "train"
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get data for a specific task and split.

        Args:
            task: Task name ("decision", "topic_type", "da")
            split: Data split ("train", "val", "test")

        Returns:
            Tuple of (texts, labels)
        """
        if self.train_df is None:
            self.split_data()

        # Get the right split
        if split == "train":
            df = self.train_df
        elif split == "val":
            df = self.val_df
        elif split == "test":
            df = self.test_df
        else:
            raise ValueError(f"Unknown split: {split}")

        # Get task config
        task_config = settings.tasks.get(task)
        if task_config is None:
            raise ValueError(f"Unknown task: {task}")

        label_column = task_config["label_column"]

        # Extract texts and labels
        texts = df["transcript"].tolist()
        labels = df[label_column].values

        # Handle label encoding
        if task == "decision":
            # Boolean to int
            labels = labels.astype(int)
        elif task == "topic_type":
            # Encode topic types
            label_mapping = {
                name: i for i, name in enumerate(task_config["class_names"])
            }
            labels = np.array([label_mapping.get(l, 3) for l in labels])
        elif task == "da":
            # Encode dialogue acts
            label_mapping = {
                name: i for i, name in enumerate(task_config["class_names"])
            }
            # Handle unknown labels
            labels = np.array([
                label_mapping.get(l, len(task_config["class_names"]) - 1)
                for l in labels
            ])

        logger.info(
            f"Task '{task}' {split} set: "
            f"{len(texts)} samples, "
            f"classes distribution: {np.bincount(labels)}"
        )

        return texts, labels

    def get_all_task_data(
        self, task: str
    ) -> Dict[str, Tuple[List[str], np.ndarray]]:
        """
        Get all splits for a task.

        Args:
            task: Task name

        Returns:
            Dictionary with train, val, test data
        """
        return {
            "train": self.get_task_data(task, "train"),
            "val": self.get_task_data(task, "val"),
            "test": self.get_task_data(task, "test"),
        }

    def get_class_weights(self, task: str) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced data.

        Args:
            task: Task name

        Returns:
            Dictionary mapping class index to weight
        """
        _, labels = self.get_task_data(task, "train")

        class_counts = np.bincount(labels)
        total = len(labels)
        n_classes = len(class_counts)

        # Balanced weights: n_samples / (n_classes * n_samples_per_class)
        weights = {
            i: total / (n_classes * count) if count > 0 else 1.0
            for i, count in enumerate(class_counts)
        }

        logger.info(f"Class weights for '{task}': {weights}")

        return weights

    def get_label_encoder(self, task: str) -> Dict[str, Any]:
        """
        Get label encoder info for a task.

        Args:
            task: Task name

        Returns:
            Dictionary with encoding info
        """
        task_config = settings.tasks.get(task)
        if task_config is None:
            raise ValueError(f"Unknown task: {task}")

        return {
            "class_names": task_config["class_names"],
            "num_classes": task_config["num_classes"],
            "name_to_idx": {
                name: i for i, name in enumerate(task_config["class_names"])
            },
            "idx_to_name": {
                i: name for i, name in enumerate(task_config["class_names"])
            },
        }

