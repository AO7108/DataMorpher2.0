# Location: app.py
# This script is the final end-to-end test for the entire pipeline.

import asyncio
import logging
import pprint

from src.utils.logger_config import setup_logging
from src.pipelines.manager import PipelineManager

async def main():
    """The main asynchronous routine for testing the full pipeline."""
    
    # 1. --- Setup ---
    # This will initialize all our components: Logger, Parser, Crawlers.
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 2. --- Instantiate the Pipeline Manager ---
    # This one line will trigger the entire setup process.
    logger.info("--- Creating PipelineManager ---")
    manager = PipelineManager()
    logger.info("--- PipelineManager Created. Ready to dispatch. ---")
    
    # 3. --- Define Test Cases ---
    # A list of raw text requests to test different parts of the pipeline.
    test_requests = [
        "get me 5 face portraits of tzuyu",        # Should trigger GoogleImagesCrawler with face filter
        "i need 3 audio clips of a cat meowing", # Should trigger YouTubeCrawler for audio
        "what is the weather like in shimla?"      # Should be parsed as 'unsupported'
    ]

    # 4. --- Run the Pipeline for each test case ---
    for request in test_requests:
        print("\n" + "="*50)
        logger.info(f"--- EXECUTING PIPELINE FOR: '{request}' ---")
        
        # This one call runs the whole process: parse -> select -> scrape
        metadata = await manager.dispatch(request)
        
        # 5. --- Display Results ---
        if metadata:
            logger.info("--- ✅ PIPELINE FINISHED SUCCESSFULLY! ---")
            logger.info("Final Returned Metadata:")
            pprint.pprint(metadata)
        else:
            logger.warning("--- ⚠️ PIPELINE FINISHED, but no dataset was produced. (This is expected for unsupported requests). ---")
        print("="*50 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger().info("Test run cancelled by user.")