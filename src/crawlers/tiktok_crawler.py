# Location: src/crawlers/tiktok_crawler.py

import logging
from pathlib import Path
from typing import List, Optional
from src.utils.models import DatasetMetadata


class TikTokCrawler:
    """
    Placeholder for TikTok. Disabled by default until API/scraper is configured.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enabled = False
        self.logger.info("TikTokCrawler disabled (no API configured).")

    async def scrape(self, subject: str, limit: int, attributes: List[str], output_dir: Path) -> Optional[DatasetMetadata]:
        if not self.enabled:
            return None
        return None

    async def scrape_with_query(self, query: str, limit: int, output_dir: Path) -> List[Path]:
        if not self.enabled:
            return []
        return []
