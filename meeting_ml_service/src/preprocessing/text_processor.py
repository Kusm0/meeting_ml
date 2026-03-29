"""
Shared text preprocessing module for all models.
"""

import re
import pickle
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger


class TextProcessor:
    """
    Shared text processor for preprocessing transcripts.
    Handles both TF-IDF vectorization and text cleaning for BERT.
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        lowercase: bool = True,
        remove_speaker_tags: bool = True,
        remove_fillers: bool = True,
    ):
        """
        Initialize text processor.

        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            lowercase: Whether to convert text to lowercase
            remove_speaker_tags: Whether to remove [A], [B], etc. tags
            remove_fillers: Whether to remove filler words (um, uh, etc.)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.remove_speaker_tags = remove_speaker_tags
        self.remove_fillers = remove_fillers

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.is_fitted = False

        # Common filler words in meeting transcripts
        self.fillers = {
            "um", "uh", "er", "ah", "mm", "hmm", "hm",
            "yeah", "yep", "okay", "ok", "right", "so",
        }

        # Speaker tag pattern [A], [B], etc.
        self.speaker_pattern = re.compile(r"\[([A-Z])\]:\s*")

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Remove speaker tags if configured
        if self.remove_speaker_tags:
            text = self.speaker_pattern.sub("", text)

        # Convert to lowercase if configured
        if self.lowercase:
            text = text.lower()

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove filler words if configured
        if self.remove_fillers:
            words = text.split()
            words = [w for w in words if w not in self.fillers]
            text = " ".join(words)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def preprocess_for_bert(self, text: str) -> str:
        """
        Preprocess text for BERT models.
        Lighter preprocessing to preserve more context.

        Args:
            text: Raw transcript text

        Returns:
            Preprocessed text for BERT
        """
        if not isinstance(text, str):
            return ""

        # Remove speaker tags but keep the structure
        text = self.speaker_pattern.sub("", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Strip
        text = text.strip()

        return text

    def fit_vectorizer(self, texts: List[str]) -> "TextProcessor":
        """
        Fit TF-IDF vectorizer on training texts.

        Args:
            texts: List of training texts

        Returns:
            Self for chaining
        """
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts...")

        # Clean texts
        cleaned_texts = [self.clean_text(t) for t in texts]

        # Initialize and fit vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            min_df=2,
            max_df=0.95,
        )

        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True

        logger.info(
            f"TF-IDF vectorizer fitted with "
            f"{len(self.vectorizer.vocabulary_)} features"
        )

        return self

    def vectorize(
        self, texts: Union[str, List[str]]
    ) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors.

        Args:
            texts: Single text or list of texts

        Returns:
            TF-IDF feature matrix
        """
        if not self.is_fitted:
            raise ValueError(
                "Vectorizer not fitted. Call fit_vectorizer first."
            )

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Clean texts
        cleaned_texts = [self.clean_text(t) for t in texts]

        # Transform
        return self.vectorizer.transform(cleaned_texts)

    def save(self, path: Path) -> None:
        """
        Save processor state to disk.

        Args:
            path: Path to save processor
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "lowercase": self.lowercase,
            "remove_speaker_tags": self.remove_speaker_tags,
            "remove_fillers": self.remove_fillers,
            "vectorizer": self.vectorizer,
            "is_fitted": self.is_fitted,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Text processor saved to {path}")

    def load(self, path: Path) -> "TextProcessor":
        """
        Load processor state from disk.

        Args:
            path: Path to load processor from

        Returns:
            Self for chaining
        """
        path = Path(path)

        with open(path, "rb") as f:
            state = pickle.load(f)

        self.max_features = state["max_features"]
        self.ngram_range = state["ngram_range"]
        self.lowercase = state["lowercase"]
        self.remove_speaker_tags = state["remove_speaker_tags"]
        self.remove_fillers = state["remove_fillers"]
        self.vectorizer = state["vectorizer"]
        self.is_fitted = state["is_fitted"]

        logger.info(f"Text processor loaded from {path}")

        return self

    def get_feature_names(self) -> List[str]:
        """
        Get feature names from TF-IDF vectorizer.

        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted.")

        return self.vectorizer.get_feature_names_out().tolist()

