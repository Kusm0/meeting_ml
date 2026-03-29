#!/usr/bin/env python
"""
Script to evaluate all trained models and generate visualizations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.config.settings import settings
from src.metrics.visualizer import MetricsVisualizer


def main():
    """Evaluate all models and generate plots."""
    logger.info("=" * 60)
    logger.info("Generating visualizations for all models")
    logger.info("=" * 60)

    visualizer = MetricsVisualizer()

    for model_type in ["tfidf", "bert"]:
        for task in ["decision", "topic_type", "da"]:
            logger.info(f"\nProcessing {model_type}/{task}...")

            try:
                figures = visualizer.load_and_visualize(model_type, task)

                if figures:
                    logger.info(
                        f"Generated {len(figures)} plots for "
                        f"{model_type}/{task}"
                    )
                else:
                    logger.warning(
                        f"No plots generated for {model_type}/{task}"
                    )

            except Exception as e:
                logger.error(
                    f"Error processing {model_type}/{task}: {e}"
                )

    logger.info("\n" + "=" * 60)
    logger.info("Visualization generation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

