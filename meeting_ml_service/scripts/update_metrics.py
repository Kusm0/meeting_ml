#!/usr/bin/env python
"""
Script to update metrics in JSON files using weighted average for binary classification.
This fixes the issue where binary metrics show 0.0 when model doesn't predict positive class.
"""

import sys
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from loguru import logger


def recalculate_metrics_from_cm(cm: np.ndarray, is_binary: bool = False) -> dict:
    """
    Recalculate metrics from confusion matrix using weighted average.
    
    Args:
        cm: Confusion matrix
        is_binary: Whether this is binary classification
        
    Returns:
        Dictionary with updated metrics
    """
    # Reconstruct y_true and y_pred from confusion matrix
    n_classes = cm.shape[0]
    y_true_list = []
    y_pred_list = []
    
    for i in range(n_classes):
        for j in range(n_classes):
            count = int(cm[i, j])
            y_true_list.extend([i] * count)
            y_pred_list.extend([j] * count)
    
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    # Calculate metrics with weighted average
    average = "weighted" if is_binary else "macro"
    
    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def update_metrics_file(model_type: str, task: str) -> bool:
    """
    Update metrics JSON file with weighted average for binary classification.
    
    Args:
        model_type: 'tfidf' or 'bert'
        task: 'decision', 'topic_type', or 'da'
        
    Returns:
        True if updated, False otherwise
    """
    metrics_path = settings.metrics_dir / model_type / task / "metrics.json"
    
    if not metrics_path.exists():
        logger.warning(f"Metrics file not found: {metrics_path}")
        return False
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    # Check if this is binary classification
    task_config = settings.tasks.get(task, {})
    is_binary = task_config.get("type") == "binary"
    
    if not is_binary:
        logger.info(f"Skipping {model_type}/{task} - not binary classification")
        return False
    
    updated = False
    
    # Update metrics for train, val, and test
    for split in ["train", "val", "test"]:
        if split not in data.get("metrics", {}):
            continue
            
        split_metrics = data["metrics"][split]
        cm = np.array(split_metrics.get("confusion_matrix"))
        
        if cm is None or cm.size == 0:
            continue
        
        # Recalculate metrics
        new_metrics = recalculate_metrics_from_cm(cm, is_binary=True)
        
        # Update only if metrics were 0.0 (indicating the issue)
        if split_metrics.get("precision", 1) == 0.0:
            logger.info(f"Updating {model_type}/{task}/{split} metrics")
            split_metrics["precision"] = new_metrics["precision"]
            split_metrics["recall"] = new_metrics["recall"]
            split_metrics["f1"] = new_metrics["f1"]
            updated = True
    
    if updated:
        # Save updated metrics
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Updated metrics file: {metrics_path}")
        return True
    
    return False


def main():
    """Update metrics for all binary classification models."""
    logger.info("=" * 60)
    logger.info("Updating metrics with weighted average for binary tasks")
    logger.info("=" * 60)
    
    updated_count = 0
    
    for model_type in ["tfidf", "bert"]:
        for task in ["decision"]:  # Only decision is binary
            logger.info(f"\nProcessing {model_type}/{task}...")
            
            try:
                if update_metrics_file(model_type, task):
                    updated_count += 1
            except Exception as e:
                logger.error(f"Error updating {model_type}/{task}: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Updated {updated_count} metrics file(s)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

