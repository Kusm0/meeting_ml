#!/usr/bin/env python
"""
Script to train all models (TF-IDF and BERT).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.training.trainer import Trainer


def main():
    """Train all models."""
    logger.info("=" * 60)
    logger.info("Starting training of all models")
    logger.info("=" * 60)

    trainer = Trainer()

    # Train TF-IDF models
    logger.info("\n" + "=" * 40)
    logger.info("Training TF-IDF models...")
    logger.info("=" * 40)

    tfidf_results = trainer.train_all_tfidf()

    for task, results in tfidf_results.items():
        test_acc = results["metrics"]["test"]["accuracy"]
        test_f1 = results["metrics"]["test"]["f1"]
        logger.info(
            f"TF-IDF {task}: accuracy={test_acc:.4f}, f1={test_f1:.4f}"
        )

    # Train BERT models
    logger.info("\n" + "=" * 40)
    logger.info("Training BERT models...")
    logger.info("=" * 40)

    bert_results = trainer.train_all_bert()

    for task, results in bert_results.items():
        test_acc = results["metrics"]["test"]["accuracy"]
        test_f1 = results["metrics"]["test"]["f1"]
        logger.info(
            f"BERT {task}: accuracy={test_acc:.4f}, f1={test_f1:.4f}"
        )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)

    logger.info("\nTF-IDF Models:")
    for task in ["decision", "topic_type", "da"]:
        acc = tfidf_results[task]["metrics"]["test"]["accuracy"]
        f1 = tfidf_results[task]["metrics"]["test"]["f1"]
        logger.info(f"  {task}: accuracy={acc:.4f}, f1={f1:.4f}")

    logger.info("\nBERT Models:")
    for task in ["decision", "topic_type", "da"]:
        acc = bert_results[task]["metrics"]["test"]["accuracy"]
        f1 = bert_results[task]["metrics"]["test"]["f1"]
        logger.info(f"  {task}: accuracy={acc:.4f}, f1={f1:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

