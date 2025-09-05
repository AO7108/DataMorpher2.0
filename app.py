# Location: app.py

import asyncio
import logging
import pprint
import argparse

from src.utils.logger_config import setup_logging
from src.pipelines.manager import PipelineManager
from src.utils.models import DatasetMetadata


async def run_pipeline_once(request: str, manager: PipelineManager, sources: dict, options: dict, logger: logging.Logger):
    """
    Executes a single pipeline run for a given user request and logs the output.

    Args:
        request (str): The user's prompt (e.g., 'create 20 pictures of tzuyu').
        manager (PipelineManager): The pipeline manager instance to dispatch the request.
        sources (dict): A dictionary of enabled data sources.
        options (dict): A dictionary of processing and training options.
        logger (logging.Logger): The logger instance for output.
    """
    final_metadata = None
    try:
        async for output in manager.dispatch(request, sources, options):
            if isinstance(output, str):
                logger.info(output)
            elif isinstance(output, DatasetMetadata):
                final_metadata = output
    except Exception as e:
        logger.error(f"--- 💥 A CRITICAL ERROR OCCURRED IN THE PIPELINE ---")
        logger.error(f"Error: {e}", exc_info=True) # exc_info=True logs the stack trace
        logger.warning("--- ⚠️ PIPELINE TERMINATED DUE TO AN ERROR. ---")
        return # Exit the function on critical error

    if final_metadata:
        logger.info("--- ✅ PIPELINE FINISHED SUCCESSFULLY! ---")
        logger.info("Final Returned Metadata:")
        pprint.pprint(final_metadata)
    else:
        logger.warning("--- ⚠️ PIPELINE FINISHED, but no dataset was produced. ---")


async def main():
    
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run DataMorpher pipeline.")
    parser.add_argument("--min-quality", choices=["low", "medium", "high"], default="medium", help="Minimum quality band.")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: enter a single prompt to run.")

    # Post-processing toggles
    parser.add_argument("--augment", action="store_true", help="Run augmentation step")
    parser.add_argument("--curate", action="store_true", help="Run CLIP-based curation step")
    parser.add_argument("--balance", action="store_true", help="Run dataset balancing step")
    parser.add_argument("--split", action="store_true", help="Split into train/val/test")
    parser.add_argument("--score", action="store_true", help="Score dataset quality and produce report")

    # Class organization (optional; parser can also trigger class mode implicitly)
    parser.add_argument("--classes-enable", action="store_true", help="Force organizing curated results into two classes")
    parser.add_argument("--class-labels", type=str, default="", help='Custom class labels as "LabelA|LabelB"')

    # Trainer
    parser.add_argument("--train", action="store_true", help="Train a simple classifier on the split dataset")
    parser.add_argument("--train-epochs", type=int, default=5, help="Training epochs (default 5)")
    parser.add_argument("--train-batch-size", type=int, default=32, help="Training batch size (default 32)")
    parser.add_argument("--train-lr", type=float, default=1e-3, help="Training learning rate (default 1e-3)")
    parser.add_argument("--train-backbone", choices=["resnet18", "mobilenet_v3_small"], default="resnet18",
                        help="Backbone architecture (default resnet18)")

    # Crawler reliability/controls
    parser.add_argument("--max-workers", type=int, default=8, help="Max parallel downloads (default 8)")
    parser.add_argument("--timeout", type=int, default=20, help="Per-download timeout seconds (default 20)")
    parser.add_argument("--curate-threshold", type=float, default=0.25, help="Curation threshold (default 0.25)")
    parser.add_argument("--balance-mode", choices=["downsample", "upsample"], default="downsample",
                        help="Balancing mode (default downsample)")

    args = parser.parse_args()

    logger.info("--- Creating PipelineManager ---")
    manager = PipelineManager()
    manager.min_quality_band = args.min_quality
    manager.crawler_overrides = {"max_workers": args.max_workers, "timeout_s": args.timeout}
    logger.info(f"Minimum quality band set to: {args.min_quality}")
    logger.info(f"Crawler settings: max_workers={args.max_workers}, timeout={args.timeout}s")
    logger.info("--- PipelineManager Created. Ready to dispatch. ---")

    # Build options
    class_labels = None
    if args.class_labels:
        parts = [p.strip() for p in args.class_labels.split("|")]
        if len(parts) == 2 and parts[0] and parts[1]:
            class_labels = parts

    sources = {"google": True}
    options = {
        "augment": args.augment,
        "curate": args.curate,
        "balance": args.balance,
        "split": args.split,
        "score": args.score,
        "train": args.train,
        "train_cfg": {
            "epochs": args.train_epochs,
            "batch_size": args.train_batch_size,
            "lr": args.train_lr,
            "backbone": args.train_backbone,
            "patience": 3,
        },
        "curate_prompt": "smiling",
        "curate_neutral": "a neutral face",
        "curate_threshold": args.curate_threshold,
        "balance_positive": "smiling",
        "balance_negative": "a neutral face",
        "balance_mode": args.balance_mode,
        # Class organization
        "organize_curated_as_classes": args.classes_enable,
        "class_labels": class_labels,
    }

    if args.interactive:
        try:
            request = input("Enter your prompt (e.g., 'create 20 pictures of tzuyu'): ").strip()
        except KeyboardInterrupt:
            logger.info("Cancelled.")
            return
        logger.info(f"--- EXECUTING PIPELINE FOR: '{request}' ---")
        await run_pipeline_once(request, manager, sources, options, logger)
        return

    # Default tests
    test_requests = [
        "create 20 pictures of tzuyu",
        "create 60 pictures of tzuyu smiling vs non smiling, equal",
        "what is the weather like in shimla?",
    ]

    for request in test_requests:
        print("\n" + "=" * 50)
        logger.info(f"--- EXECUTING PIPELINE FOR: '{request}' ---")
        await run_pipeline_once(request, manager, sources, options, logger)
        print("=" * 50 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger().info("Test run cancelled by user.")
