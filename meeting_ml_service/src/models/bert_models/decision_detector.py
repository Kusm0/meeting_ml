"""
BERT-based Decision Detection model.
Binary classification using fine-tuned DistilBERT.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm
from loguru import logger

from src.models.base_model import BaseModel
from src.config.settings import settings


class MeetingDataset(Dataset):
    """Dataset for meeting transcripts."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[np.ndarray],
        tokenizer: DistilBertTokenizer,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


class DecisionDetectorBERTModel(nn.Module):
    """BERT-based binary classifier."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class DecisionDetectorBERT(BaseModel):
    """
    DistilBERT fine-tuned for Decision Detection.
    Binary classification task.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 256,  # Reduced for CPU memory
        batch_size: int = 4,    # Reduced for CPU memory
        learning_rate: float = 2e-5,
        epochs: int = 3,        # Reduced for faster training
        dropout: float = 0.1,
        warmup_steps: int = 50,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
    ):
        """
        Initialize BERT Decision Detector.

        Args:
            model_name: Pretrained model name
            max_length: Maximum sequence length
            batch_size: Training batch size
            learning_rate: Learning rate
            epochs: Number of training epochs
            dropout: Dropout rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for optimizer
            device: Device to use (cuda/cpu)
        """
        task_config = settings.tasks["decision"]
        super().__init__(
            task="decision",
            model_type="bert",
            num_classes=task_config["num_classes"],
            class_names=task_config["class_names"],
        )

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dropout = dropout
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None  # Initialized during training

        logger.info(f"Using device: {self.device}")

    def _init_model(self) -> None:
        """Initialize the BERT model."""
        self.model = DecisionDetectorBERTModel(
            model_name=self.model_name,
            dropout=self.dropout,
        )
        self.model.to(self.device)

    def train(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training texts
            y_train: Training labels
            X_val: Validation texts
            y_val: Validation labels

        Returns:
            Training history
        """
        logger.info(
            f"Training Decision Detector (BERT) on "
            f"{len(X_train)} samples..."
        )

        # Initialize model
        self._init_model()

        # Create datasets
        train_dataset = MeetingDataset(
            X_train, y_train, self.tokenizer, self.max_length
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = MeetingDataset(
                X_val, y_val, self.tokenizer, self.max_length
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        class_weights = torch.tensor(
            [len(y_train) / (2 * c) for c in class_counts],
            dtype=torch.float32,
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )

        # Training history
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
            )

            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                progress.set_postfix(
                    loss=loss.item(),
                    acc=correct / total,
                )

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}"
                )

            # Save checkpoint after each epoch
            self._save_checkpoint(epoch, history)

        self.is_trained = True
        return history

    def _save_checkpoint(self, epoch: int, history: Dict[str, Any]) -> None:
        """Save training checkpoint."""
        from src.config.settings import settings
        
        checkpoint_dir = settings.base_dir / "checkpoints" / "bert" / "decision"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "history": history,
            "config": {
                "model_name": self.model_name,
                "max_length": self.max_length,
                "dropout": self.dropout,
            }
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as latest
        latest_path = checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(data_loader), correct / total

    def predict(self, X: List[str]) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input texts

        Returns:
            Predicted class indices
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input texts

        Returns:
            Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        self.model.eval()

        dataset = MeetingDataset(X, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = path / "model.pt"
        torch.save(self.model.state_dict(), model_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(path / "tokenizer")

        # Save config
        config = {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "dropout": self.dropout,
            "is_trained": self.is_trained,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        self.model_name = config["model_name"]
        self.max_length = config["max_length"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]
        self.dropout = config["dropout"]
        self.is_trained = config["is_trained"]

        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            path / "tokenizer"
        )

        # Initialize and load model
        self._init_model()
        self.model.load_state_dict(
            torch.load(path / "model.pt", map_location=self.device)
        )
        self.model.eval()

        logger.info(f"Model loaded from {path}")

