"""
Training API routes.
"""

import asyncio
from typing import Any, Dict, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from loguru import logger

from src.training.trainer import Trainer
from src.training.data_loader import DataLoader as DataLoaderClass

router = APIRouter(prefix="/training", tags=["training"])


# Store for training jobs
training_jobs: Dict[str, Dict[str, Any]] = {}
job_counter = 0


class TrainRequest(BaseModel):
    """Request model for training."""

    model_type: str = Field(
        ...,
        description="Model type: 'tfidf' or 'bert'"
    )
    task: str = Field(
        ...,
        description="Task: 'decision', 'topic_type', or 'da'"
    )
    params: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional model parameters"
    )


class TrainAllRequest(BaseModel):
    """Request model for training all models."""

    model_type: Optional[str] = Field(
        None,
        description="Model type: 'tfidf', 'bert', or None for all"
    )


class TrainResponse(BaseModel):
    """Response model for training."""

    status: str
    job_id: str
    message: str


def run_training(
    job_id: str,
    model_type: str,
    task: str,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    """Run training in background."""
    global training_jobs

    try:
        training_jobs[job_id]["status"] = "running"

        trainer = Trainer()
        results = trainer.train_model(
            model_type,
            task,
            **(params or {}),
        )

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["results"] = {
            "test_accuracy": results["metrics"]["test"]["accuracy"],
            "test_f1": results["metrics"]["test"]["f1"],
        }

    except Exception as e:
        logger.error(f"Training error: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


def run_training_all(
    job_id: str,
    model_type: Optional[str] = None,
) -> None:
    """Run training for all models in background."""
    global training_jobs

    try:
        training_jobs[job_id]["status"] = "running"

        trainer = Trainer()

        if model_type is None:
            results = trainer.train_all()
        elif model_type == "tfidf":
            results = {"tfidf": trainer.train_all_tfidf()}
        elif model_type == "bert":
            results = {"bert": trainer.train_all_bert()}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Extract summary
        summary = {}
        for mt, tasks in results.items():
            summary[mt] = {}
            for task, task_results in tasks.items():
                summary[mt][task] = {
                    "test_accuracy": task_results["metrics"]["test"]["accuracy"],
                    "test_f1": task_results["metrics"]["test"]["f1"],
                }

        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["results"] = summary

    except Exception as e:
        logger.error(f"Training error: {e}")
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """
    Start training a model.

    Args:
        request: Training request
        background_tasks: FastAPI background tasks

    Returns:
        Training job info
    """
    global job_counter, training_jobs

    # Validate inputs
    if request.model_type not in ["tfidf", "bert"]:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'tfidf' or 'bert'"
        )

    if request.task not in ["decision", "topic_type", "da"]:
        raise HTTPException(
            status_code=400,
            detail="task must be 'decision', 'topic_type', or 'da'"
        )

    # Create job
    job_counter += 1
    job_id = f"job_{job_counter}"

    training_jobs[job_id] = {
        "status": "pending",
        "model_type": request.model_type,
        "task": request.task,
    }

    # Start training in background
    background_tasks.add_task(
        run_training,
        job_id,
        request.model_type,
        request.task,
        request.params,
    )

    return {
        "status": "started",
        "job_id": job_id,
        "message": f"Training {request.model_type}/{request.task} started",
    }


@router.post("/train-all", response_model=TrainResponse)
async def train_all_models(
    request: TrainAllRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    """
    Start training all models.

    Args:
        request: Training request
        background_tasks: FastAPI background tasks

    Returns:
        Training job info
    """
    global job_counter, training_jobs

    # Validate inputs
    if request.model_type is not None and \
       request.model_type not in ["tfidf", "bert"]:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'tfidf', 'bert', or null"
        )

    # Create job
    job_counter += 1
    job_id = f"job_{job_counter}"

    training_jobs[job_id] = {
        "status": "pending",
        "model_type": request.model_type or "all",
    }

    # Start training in background
    background_tasks.add_task(
        run_training_all,
        job_id,
        request.model_type,
    )

    model_desc = request.model_type or "all"
    return {
        "status": "started",
        "job_id": job_id,
        "message": f"Training {model_desc} models started",
    }


@router.get("/status/{job_id}")
async def get_training_status(job_id: str) -> Dict[str, Any]:
    """
    Get status of a training job.

    Args:
        job_id: Training job ID

    Returns:
        Job status
    """
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return training_jobs[job_id]


@router.get("/jobs")
async def list_training_jobs() -> Dict[str, Dict[str, Any]]:
    """
    List all training jobs.

    Returns:
        All training jobs
    """
    return training_jobs

