"""
Metrics API routes.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from loguru import logger

from src.config.settings import settings

router = APIRouter(prefix="/metrics", tags=["metrics"])


def load_metrics(model_type: str, task: str) -> Optional[Dict[str, Any]]:
    """Load metrics from file."""
    metrics_path = settings.metrics_dir / model_type / task / "metrics.json"

    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        return json.load(f)


def load_history(model_type: str, task: str) -> Optional[Dict[str, Any]]:
    """Load training history from file."""
    history_path = (
        settings.metrics_dir / model_type / task / "training_history.json"
    )

    if not history_path.exists():
        return None

    with open(history_path, "r") as f:
        return json.load(f)


@router.get("/{model_type}/{task}")
async def get_model_metrics(
    model_type: str,
    task: str,
) -> Dict[str, Any]:
    """
    Get metrics for a specific model.

    Args:
        model_type: 'tfidf' or 'bert'
        task: 'decision', 'topic_type', or 'da'

    Returns:
        Model metrics
    """
    if model_type not in ["tfidf", "bert"]:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'tfidf' or 'bert'"
        )

    if task not in ["decision", "topic_type", "da"]:
        raise HTTPException(
            status_code=400,
            detail="task must be 'decision', 'topic_type', or 'da'"
        )

    metrics = load_metrics(model_type, task)
    history = load_history(model_type, task)

    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for {model_type}/{task}"
        )

    return {
        "model_type": model_type,
        "task": task,
        "metrics": metrics.get("metrics", {}),
        "history": history or {},
    }


@router.get("/all")
async def get_all_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Get metrics for all models.

    Returns:
        All model metrics
    """
    all_metrics = {}

    for model_type in ["tfidf", "bert"]:
        all_metrics[model_type] = {}

        for task in ["decision", "topic_type", "da"]:
            metrics = load_metrics(model_type, task)

            if metrics is not None:
                all_metrics[model_type][task] = {
                    "test": metrics.get("metrics", {}).get("test", {}),
                }
            else:
                all_metrics[model_type][task] = None

    return all_metrics


@router.get("/comparison")
async def get_metrics_comparison() -> Dict[str, Any]:
    """
    Get comparison of TF-IDF vs BERT metrics for all tasks.

    Returns:
        Metrics comparison
    """
    comparison = {}

    for task in ["decision", "topic_type", "da"]:
        comparison[task] = {}

        for model_type in ["tfidf", "bert"]:
            metrics = load_metrics(model_type, task)

            if metrics is not None:
                test_metrics = metrics.get("metrics", {}).get("test", {})
                comparison[task][model_type] = {
                    "accuracy": test_metrics.get("accuracy"),
                    "precision": test_metrics.get("precision"),
                    "recall": test_metrics.get("recall"),
                    "f1": test_metrics.get("f1"),
                }
            else:
                comparison[task][model_type] = None

    return comparison


@router.get("/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get summary of all model metrics.

    Returns:
        Summary table data
    """
    summary = []

    for model_type in ["tfidf", "bert"]:
        for task in ["decision", "topic_type", "da"]:
            metrics = load_metrics(model_type, task)

            if metrics is not None:
                test_metrics = metrics.get("metrics", {}).get("test", {})
                summary.append({
                    "model_type": model_type.upper(),
                    "task": task,
                    "accuracy": test_metrics.get("accuracy"),
                    "precision": test_metrics.get("precision"),
                    "recall": test_metrics.get("recall"),
                    "f1": test_metrics.get("f1"),
                })
            else:
                summary.append({
                    "model_type": model_type.upper(),
                    "task": task,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                })

    return {"summary": summary}


@router.get("/plots/{model_type}/{task}")
async def get_available_plots(
    model_type: str,
    task: str,
) -> Dict[str, List[str]]:
    """
    Get list of available plots for a model.

    Args:
        model_type: 'tfidf' or 'bert'
        task: 'decision', 'topic_type', or 'da'

    Returns:
        List of plot file names
    """
    plots_dir = settings.metrics_dir / model_type / task / "plots"

    if not plots_dir.exists():
        return {"plots": []}

    plots = [f.name for f in plots_dir.glob("*.png")]

    return {"plots": plots}

