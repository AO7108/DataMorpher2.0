# Location: src/crawlers/twitter_crawler.py

import logging
from pathlib import Path
from typing import List, Optional
from src.utils.models import DatasetMetadata
import os


class TwitterCrawler:
    """
    Placeholder crawler for Twitter/X.
    Requires TWITTER_BEARER_TOKEN in the environment to enable.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.token = os.getenv("TWITTER_BEARER_TOKEN")
        if not self.token:
            self.logger.info("TwitterCrawler disabled: missing TWITTER_BEARER_TOKEN.")
            self.enabled = False
            return
        self.enabled = True
        self.max_results = 20

    async def scrape(self, subject: str, limit: int, attributes: List[str], output_dir: Path) -> Optional[DatasetMetadata]:
        if not self.enabled:
            return None
        self.logger.info("TwitterCrawler enabled but not implemented yet; returning None.")
        return None

    async def scrape_with_query(self, query: str, limit: int, output_dir: Path) -> List[Path]:
        if not self.enabled:
            return []
        self.logger.info("TwitterCrawler scrape_with_query not implemented; returning [].")
        return []
