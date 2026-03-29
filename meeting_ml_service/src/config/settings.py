"""
Application settings and configuration.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    name: str
    task: str  # "decision", "topic_type", "da"
    model_type: str  # "tfidf", "bert"


@dataclass
class TFIDFConfig:
    """TF-IDF model configuration."""

    max_features: int = 10000
    ngram_range: tuple = (1, 2)
    max_iter: int = 1000
    class_weight: str = "balanced"


@dataclass
class BERTConfig:
    """BERT model configuration."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = 256  # Reduced from 512 for memory
    batch_size: int = 4    # Reduced from 16 for memory (CPU)
    learning_rate: float = 2e-5
    epochs: int = 3        # Reduced from 5 for faster training
    dropout: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    save_checkpoints: bool = True  # Save model checkpoints


@dataclass
class TrainingConfig:
    """Training configuration."""

    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_state: int = 42
    early_stopping_patience: int = 3


@dataclass
class Settings:
    """Application settings."""

    # Paths
    base_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def models_dir(self) -> Path:
        return self.base_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.base_dir / "metrics"

    @property
    def dataset_path(self) -> Path:
        # Check environment variable first (for Docker)
        env_path = os.environ.get("DATASET_PATH")
        if env_path:
            return Path(env_path)
        
        # Check if running in Docker (datasets mounted at /app/datasets)
        docker_path = self.base_dir / "datasets" / "ami_topic_dataset_extended.csv"
        if docker_path.exists():
            return docker_path
        
        # Fallback to local development path
        return (
            self.base_dir.parent / "datasets" / "ami_topic_dataset_extended.csv"
        )

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Gradio settings
    gradio_host: str = "0.0.0.0"
    gradio_port: int = 7860

    # Model configs
    tfidf_config: TFIDFConfig = field(default_factory=TFIDFConfig)
    bert_config: BERTConfig = field(default_factory=BERTConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # Task definitions
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "decision": {
            "name": "Decision Detection",
            "type": "binary",
            "label_column": "has_decision_annotation",
            "num_classes": 2,
            "class_names": ["No Decision", "Has Decision"],
        },
        "topic_type": {
            "name": "Topic Type Classification",
            "type": "multiclass",
            "label_column": "topic_type",
            "num_classes": 4,
            "class_names": ["Top level", "Sub-topics", "Functional", "other"],
        },
        "da": {
            "name": "Dialogue Act Classification",
            "type": "multiclass",
            "label_column": "da_dominant",
            "num_classes": 15,
            "class_names": [
                "Inform", "Assess", "Suggest", "Fragment", "Backchannel",
                "Stall", "Elicit-Inform", "Elicit-Assessment",
                "Elicit-Offer-Or-Suggestion", "Comment-About-Understanding",
                "Be-Positive", "Be-Negative", "Offer", "Other", "Unknown"
            ],
        },
    })

    def __post_init__(self):
        """Create necessary directories."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for models
        for model_type in ["tfidf", "bert"]:
            for task in ["decision", "topic_type", "da"]:
                (self.models_dir / model_type / task).mkdir(
                    parents=True, exist_ok=True
                )
                (self.metrics_dir / model_type / task).mkdir(
                    parents=True, exist_ok=True
                )


# Global settings instance
settings = Settings()

