#!/usr/bin/env python3
"""
Simple script to update metrics using weighted average from per_class metrics.
No external dependencies required - uses only standard library.
"""

import json
from pathlib import Path


def calculate_weighted_from_per_class(per_class: dict) -> dict:
    """
    Calculate weighted metrics from per_class metrics.
    
    Args:
        per_class: Dictionary with per-class metrics
        
    Returns:
        Dictionary with weighted precision, recall, f1
    """
    total_support = sum(m.get('support', 0) for m in per_class.values())
    
    if total_support == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    weighted_precision = sum(
        m.get('precision', 0) * m.get('support', 0) 
        for m in per_class.values()
    ) / total_support
    
    weighted_recall = sum(
        m.get('recall', 0) * m.get('support', 0) 
        for m in per_class.values()
    ) / total_support
    
    weighted_f1 = sum(
        m.get('f1', 0) * m.get('support', 0) 
        for m in per_class.values()
    ) / total_support
    
    return {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
    }


def update_metrics_file(metrics_path: Path) -> bool:
    """Update metrics file with weighted averages."""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    updated = False
    
    # Update metrics for train, val, and test
    for split in ["train", "val", "test"]:
        if split not in data.get("metrics", {}):
            continue
            
        split_metrics = data["metrics"][split]
        per_class = split_metrics.get("per_class", {})
        
        # Only update if precision is 0.0 (indicating binary average issue)
        if split_metrics.get("precision", 1) == 0.0 and per_class:
            new_metrics = calculate_weighted_from_per_class(per_class)
            split_metrics["precision"] = new_metrics["precision"]
            split_metrics["recall"] = new_metrics["recall"]
            split_metrics["f1"] = new_metrics["f1"]
            updated = True
            print(f"Updated {split} metrics: precision={new_metrics['precision']:.4f}, "
                  f"recall={new_metrics['recall']:.4f}, f1={new_metrics['f1']:.4f}")
    
    if updated:
        # Save updated metrics
        with open(metrics_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nUpdated metrics file: {metrics_path}")
        return True
    
    return False


def main():
    """Update metrics for BERT decision model."""
    # Path relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    metrics_path = project_root / "metrics" / "bert" / "decision" / "metrics.json"
    
    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}")
        return
    
    print("Updating BERT decision metrics with weighted average...")
    update_metrics_file(metrics_path)
    print("Done!")


if __name__ == "__main__":
    main()





