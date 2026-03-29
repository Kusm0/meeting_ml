"""
FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.config.settings import settings
from src.api.routes import inference_router, training_router, metrics_router


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="Meeting ML Service",
        description=(
            "ML service for meeting transcript analysis: "
            "Decision Detection, Topic Type Classification, "
            "Dialogue Act Classification"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(inference_router, prefix="/api")
    app.include_router(training_router, prefix="/api")
    app.include_router(metrics_router, prefix="/api")

    # Static files for plots
    plots_dir = settings.metrics_dir
    if plots_dir.exists():
        app.mount(
            "/static/metrics",
            StaticFiles(directory=str(plots_dir)),
            name="metrics",
        )

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "Meeting ML Service",
            "version": "1.0.0",
            "docs": "/docs",
            "tasks": ["decision", "topic_type", "da"],
            "models": ["tfidf", "bert"],
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.on_event("startup")
    async def startup():
        """Startup event handler."""
        logger.info("Starting Meeting ML Service...")

        # Ensure directories exist
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        settings.metrics_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Service started successfully")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

