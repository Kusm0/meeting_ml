"""
Inference API routes.
"""

from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.inference.predictor import get_predictor

router = APIRouter(prefix="/inference", tags=["inference"])


class InferenceRequest(BaseModel):
    """Request model for inference."""

    transcript: str = Field(..., description="Meeting transcript text")
    model_type: Optional[str] = Field(
        "tfidf",
        description="Model type: 'tfidf', 'bert', or 'both'"
    )


class PredictionResult(BaseModel):
    """Single prediction result."""

    prediction: Optional[str] = None
    prediction_idx: Optional[int] = None
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    model_type: Optional[str] = None
    task: Optional[str] = None
    error: Optional[str] = None


class AllPredictionsResult(BaseModel):
    """All predictions result."""

    decision: PredictionResult
    topic_type: PredictionResult
    da: PredictionResult


@router.post("/decision", response_model=PredictionResult)
async def predict_decision(request: InferenceRequest) -> Dict[str, Any]:
    """
    Predict whether transcript contains a decision.

    Args:
        request: Inference request with transcript

    Returns:
        Prediction result
    """
    predictor = get_predictor()

    model_type = request.model_type
    if model_type == "both":
        model_type = "tfidf"  # Default to tfidf for single endpoint

    result = predictor.predict_single(
        request.transcript,
        model_type,
        "decision",
    )

    return result


@router.post("/topic-type", response_model=PredictionResult)
async def predict_topic_type(request: InferenceRequest) -> Dict[str, Any]:
    """
    Predict topic type of transcript.

    Args:
        request: Inference request with transcript

    Returns:
        Prediction result
    """
    predictor = get_predictor()

    model_type = request.model_type
    if model_type == "both":
        model_type = "tfidf"

    result = predictor.predict_single(
        request.transcript,
        model_type,
        "topic_type",
    )

    return result


@router.post("/da", response_model=PredictionResult)
async def predict_da(request: InferenceRequest) -> Dict[str, Any]:
    """
    Predict dominant dialogue act of transcript.

    Args:
        request: Inference request with transcript

    Returns:
        Prediction result
    """
    predictor = get_predictor()

    model_type = request.model_type
    if model_type == "both":
        model_type = "tfidf"

    result = predictor.predict_single(
        request.transcript,
        model_type,
        "da",
    )

    return result


@router.post("/all")
async def predict_all(request: InferenceRequest) -> Dict[str, Any]:
    """
    Predict all tasks with specified model type.

    Args:
        request: Inference request with transcript and model type

    Returns:
        All predictions
    """
    predictor = get_predictor()

    model_type = request.model_type

    if model_type == "both":
        # Return predictions from both model types
        return predictor.predict_all_models(request.transcript)
    else:
        # Return predictions from specified model type
        return {
            model_type: predictor.predict_all_tasks(
                request.transcript, model_type
            )
        }


@router.get("/status")
async def get_model_status() -> Dict[str, Any]:
    """
    Get status of all loaded models.

    Returns:
        Model status dictionary
    """
    predictor = get_predictor()
    return predictor.get_model_status()

