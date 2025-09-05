import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import io

import aiohttp
from PIL import Image, UnidentifiedImageError

from src.utils.models import DatasetMetadata
from src.utils import config_loader

# Dependency check
try:
    import praw  # type: ignore
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False


class RedditCrawler:
    """
    A crawler for fetching images from Reddit using the PRAW library.
    - Searches relevant subreddits for image posts matching a query.
    - Downloads images asynchronously and converts them to a standard format.
    - Disabled if PRAW is not installed or if Reddit API credentials are not in the .env file.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client_id = config_loader.get_reddit_client_id()
        self.client_secret = config_loader.get_reddit_client_secret()
        self.user_agent = config_loader.get_reddit_user_agent()

        if not HAS_PRAW:
            self.logger.warning("RedditCrawler is disabled: 'praw' library not found. Please install it.")
            self.enabled = False
            return

        if not (self.client_id and self.client_secret and self.user_agent):
            self.logger.warning("RedditCrawler is disabled: Missing Reddit API credentials in .env file.")
            self.enabled = False
            return

        try:
            # PRAW is synchronous, so we initialize it here.
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                read_only=True, # Explicitly set to read-only mode
            )
            self.enabled = True
            self.logger.info("RedditCrawler initialized successfully.")
        except Exception as e:
            self.logger.error(f"RedditCrawler initialization failed: {e}", exc_info=True)
            self.enabled = False

        # Tunables (can be overridden by PipelineManager)
        self.max_workers = 8
        self.timeout_s = 20
        self.max_retries = 2
        self._image_exts = (".jpg", ".jpeg", ".png", ".webp")
        
        # We can restrict to known good image hosts if needed, but we'll allow all for now
        self.allowed_hosts: set[str] = {"i.redd.it", "i.imgur.com"}

    async def scrape(self, subject: str, limit: int, attributes: List[str], output_dir: Path) -> Optional[DatasetMetadata]:
        if not self.enabled:
            return None

        query = self._build_query(subject, attributes)
        self.logger.info(f"Searching Reddit with query: '{query}'")
        
        urls = await self._search_image_urls(query, limit)
        if not urls:
            self.logger.warning("No image URLs found on Reddit for the query.")
            return None

        files = await self._download_images(urls, output_dir, subject)
        if not files:
            self.logger.warning("Failed to download any images from the found Reddit URLs.")
            return None
            
        self.logger.info(f"Reddit crawl successful. Created a dataset of {len(files)} image files.")
        return DatasetMetadata(
            subject=subject,
            modality="image",
            source="reddit",
            attributes=attributes or [],
            files=files,
        )

    async def scrape_with_query(self, query: str, limit: int, output_dir: Path) -> List[Path]:
        if not self.enabled:
            return []
        
        urls = await self._search_image_urls(query, limit)
        if not urls:
            return []
            
        files = await self._download_images(urls, output_dir, subject="recrawl")
        return files

    def _build_query(self, subject: str, attributes: List[str]) -> str:
        parts = [subject] + (attributes or [])
        return " ".join([p for p in parts if p]).strip()

    def _sync_search_logic(self, query: str, limit: int) -> List[str]:
        """Synchronous helper function that contains the PRAW logic."""
        urls: List[str] = []
        seen_urls = set()

        # Define a search strategy: search specific subreddits first, then all of Reddit
        search_subreddits = ["pics", "itookapicture", "art", "all"]
        
        for sub_name in search_subreddits:
            if len(urls) >= limit:
                break
            try:
                subreddit = self.reddit.subreddit(sub_name)
                # PRAW's search is a generator, limit it to avoid excessive API calls
                submissions = subreddit.search(query, sort="relevance", limit=limit * 2)
                
                for post in submissions:
                    if len(urls) >= limit:
                        break
                    # Filter out NSFW posts and posts without a valid URL
                    if getattr(post, "over_18", False):
                        continue
                    url = getattr(post, "url", "")
                    if url and url not in seen_urls and self._is_image_url(url):
                        urls.append(url)
                        seen_urls.add(url)

            except Exception as e:
                self.logger.warning(f"Error searching subreddit '{sub_name}': {e}")
                continue
        
        return urls[:limit]

    async def _search_image_urls(self, query: str, limit: int) -> List[str]:
        """Asynchronously runs the synchronous PRAW search logic in a separate thread."""
        try:
            # This runs the blocking PRAW code in a thread pool, keeping the app responsive
            loop = asyncio.get_running_loop()
            urls = await loop.run_in_executor(None, self._sync_search_logic, query, limit)
            self.logger.info(f"Found {len(urls)} potential image URLs on Reddit.")
            return urls
        except Exception as e:
            self.logger.error(f"An error occurred during Reddit search execution: {e}", exc_info=True)
            return []

    def _is_image_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            # Restrict to allowed hosts and valid image extensions
            if self.allowed_hosts and parsed.netloc not in self.allowed_hosts:
                return False
            return url.lower().endswith(self._image_exts)
        except Exception:
            return False

    async def _download_images(self, urls: List[str], output_dir: Path, subject: str) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        sem = asyncio.Semaphore(self.max_workers)
        headers = {
            "Accept": "image/*,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (compatible; DataMorpher/2.0)",
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            tasks = [self._download_one(session, sem, url, idx, output_dir, subject) for idx, url in enumerate(urls)]
            results = await asyncio.gather(*tasks)
        
        return [path for path in results if path is not None]

    async def _download_one(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, url: str, idx: int, output_dir: Path, subject: str) -> Optional[Path]:
        async with sem:
            for attempt in range(self.max_retries):
                try:
                    async with session.get(url, timeout=self.timeout_s) as resp:
                        if resp.status != 200:
                            self.logger.warning(f"Got status {resp.status} for {url} on attempt {attempt+1}")
                            await asyncio.sleep(0.5 * (attempt + 1)) # Backoff
                            continue
                        
                        content_type = resp.headers.get("Content-Type", "").lower()
                        if not content_type.startswith("image/"):
                            self.logger.warning(f"Skipping non-image URL {url} (Content-Type: {content_type})")
                            return None

                        data = await resp.read()
                        
                        # Verify and standardize the image
                        with Image.open(io.BytesIO(data)) as img:
                            img = img.convert("RGB")
                            filename = f"reddit_{subject.replace(' ', '_')}_{idx:04d}.jpg"
                            filepath = output_dir / filename
                            img.save(filepath, "JPEG", quality=90)
                            return filepath

                except UnidentifiedImageError:
                    self.logger.warning(f"Could not identify image from URL: {url}. Skipping.")
                    return None
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout downloading {url} on attempt {attempt+1}")
                except Exception as e:
                    self.logger.error(f"Error downloading {url}: {e}", exc_info=False) # Keep logs clean
                    break # Break on unexpected errors
            
            return None