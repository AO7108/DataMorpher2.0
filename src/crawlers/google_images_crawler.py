# Location: src/crawlers/google_images_crawler.py

import io
import logging
import asyncio
from pathlib import Path
from urllib.parse import urlparse
import random
import time

import aiohttp
import aiofiles  # kept
from serpapi import GoogleSearch
from PIL import Image

# face_recognition is optional; we handle fallback gracefully
try:
    import face_recognition  # type: ignore
    FACE_REC_AVAILABLE = True
except Exception:
    face_recognition = None  # type: ignore
    FACE_REC_AVAILABLE = False

from src.utils.models import DatasetMetadata
from src.utils import config_loader


class GoogleImagesCrawler:
    """
    A crawler for fetching images from Google Images using the SerpApi service.
    """

    # Minimum fraction of image area a detected face must cover to be considered valid (when "face" requested)
    FACE_MIN_AREA_RATIO = 0.02

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = config_loader.get_serpapi_key()
        if not self.api_key:
            self.logger.warning("SerpApi API key not found. GoogleImagesCrawler will be non-functional.")
        else:
            self.logger.info("GoogleImagesCrawler initialized successfully with API key.")

        # Tunables (overridable by PipelineManager)
        self.max_workers = 8
        self.timeout_s = 20
        self.max_retries = 3
        self.per_host_rps = 1.5
        self._last_host_call: dict[str, float] = {}

        # Known non-image hosts to skip early
        self._blocked_hosts = {
            "lookaside.fbsbx.com",
            "lookaside.instagram.com",
            "www.tiktok.com",
            "tiktok.com",
        }

    async def scrape(
        self,
        subject: str,
        limit: int,
        attributes: list[str],
        output_dir: Path,
    ) -> DatasetMetadata | None:
        """
        Main crawl for the subject using derived query from subject + attributes.
        """
        self.logger.info(f"Starting Google Images crawl for '{subject}' with attributes {attributes}.")
        if not self.api_key:
            self.logger.error("Cannot scrape: SerpApi key is not configured.")
            return None

        query = self._build_search_query(subject, attributes)
        image_results = await self._get_image_urls(query, limit)
        if not image_results:
            self.logger.warning(f"No image results found for query: '{query}'. Aborting crawl.")
            return None

        downloaded_paths = await self._download_images_async(image_results, output_dir, subject)
        if not downloaded_paths:
            self.logger.warning("Failed to download any images. Aborting crawl.")
            return None

        final_paths: list[Path] = []
        wants_face = any(attr.lower() == "face" for attr in attributes)

        # Only apply human face filtering if explicitly requested via "face"
        if wants_face:
            if not FACE_REC_AVAILABLE:
                self.logger.warning("face_recognition not available. Skipping face filtering.")
                final_paths = downloaded_paths
            else:
                self.logger.info("'face' attribute detected. Applying human face filtering...")
                for path in downloaded_paths:
                    if self._filter_image_for_face(path):
                        final_paths.append(path)
                self.logger.info(
                    f"Filtering complete. {len(final_paths)} of {len(downloaded_paths)} images contained valid human faces."
                )
        else:
            final_paths = downloaded_paths

        if not final_paths:
            self.logger.warning("No images remained after download/filtering. Aborting crawl.")
            return None

        self.logger.info(f"Crawl successful. Creating metadata for {len(final_paths)} images.")
        metadata = DatasetMetadata(
            subject=subject,
            modality="image",
            source="google_images",
            attributes=attributes,
            files=final_paths,
        )
        return metadata

    async def scrape_with_query(
        self,
        query: str,
        limit: int,
        output_dir: Path,
    ) -> list[Path]:
        """
        Internal helper for minority-class recrawl: fetch images using a custom query string.
        Returns a list of downloaded file paths (standardized JPEGs).
        """
        if not self.api_key:
            self.logger.error("Cannot scrape_with_query: SerpApi key is not configured.")
            return []
        self.logger.info(f"[Recrawl] Query override: '{query}' (limit={limit})")
        image_results = await self._get_image_urls(query, limit)
        if not image_results:
            self.logger.warning(f"[Recrawl] No image results found for query: '{query}'.")
            return []
        # Use "recrawl" as the subject marker for filenames; callers can rename/merge later as needed
        downloaded_paths = await self._download_images_async(image_results, output_dir, subject="recrawl")
        return downloaded_paths

    def _build_search_query(self, subject: str, attributes: list[str]) -> str:
        self.logger.debug(f"Building search query for subject='{subject}' and attributes={attributes}")
        query_parts = [subject] + attributes
        full_query = " ".join(query_parts)
        self.logger.info(f"Constructed search query: '{full_query}'")
        return full_query

    async def _get_image_urls(self, query: str, limit: int) -> list[dict]:
        if not self.api_key:
            self.logger.error("SerpApi API key not found. Cannot fetch image URLs.")
            return []

        all_image_results: list[dict] = []
        page_num = 0

        try:
            while len(all_image_results) < limit:
                params = {"api_key": self.api_key, "engine": "google_images", "q": query, "ijn": page_num}
                self.logger.debug(f"Querying SerpApi on page {page_num}...")
                search = GoogleSearch(params)

                loop = asyncio.get_running_loop()
                search_results = await loop.run_in_executor(None, search.get_dict)

                if "images_results" not in search_results:
                    self.logger.warning("No 'images_results' in API response.")
                    break

                page_results = search_results["images_results"]
                all_image_results.extend(page_results)
                self.logger.info(f"Found {len(page_results)} images on page {page_num}. Total found: {len(all_image_results)}.")

                if "serpapi_pagination" not in search_results:
                    break
                page_num += 1

            final_results = all_image_results[:limit]
            self.logger.info(f"Finished search. Total unique image results gathered: {len(final_results)}.")
            return final_results

        except Exception as e:
            self.logger.error(f"An error occurred while calling SerpApi: {e}")
            return []

    def _pick_best_image_url(self, item: dict) -> str | None:
        candidates = []
        for key in ("original", "thumbnail", "source", "link"):
            val = item.get(key)
            if isinstance(val, str) and val:
                candidates.append(val)
        # Prefer direct image URLs and non-blocked hosts
        for url in candidates:
            host = urlparse(url).netloc.lower()
            if host in self._blocked_hosts:
                continue
            if any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                return url
        for url in candidates:
            host = urlparse(url).netloc.lower()
            if host not in self._blocked_hosts:
                return url
        return None

    # ---------- helpers for rate limiting and backoff ----------
    def _rate_limit(self, url: str):
        host = urlparse(url).netloc
        min_interval = 1.0 / self.per_host_rps
        now = time.time()
        last = self._last_host_call.get(host, 0.0)
        delta = now - last
        if delta < min_interval:
            wait_s = min_interval - delta
            return wait_s
        return 0.0

    async def _sleep_with_jitter(self, base: float):
        if base <= 0:
            return
        await asyncio.sleep(base * (1.0 + random.random() * 0.25))
    # ---------------------------------------------------------------

    async def _download_images_async(self, image_results: list[dict], output_dir: Path, subject: str) -> list[Path]:
        self.logger.info(f"Starting async download of up to {len(image_results)} images to '{output_dir}'")
        output_dir.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(self.max_workers)

        headers = {
            "Accept": "image/*,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (compatible; DataMorpher/2.0)",
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            async def guarded(result: dict, idx: int):
                async with sem:
                    return await self._download_single_image(session, result, idx, output_dir, subject)

            tasks = [asyncio.create_task(guarded(result, i)) for i, result in enumerate(image_results)]
            downloaded_paths = await asyncio.gather(*tasks)
            successful_paths = [path for path in downloaded_paths if path is not None]
            self.logger.info(f"Finished download process. Successfully downloaded {len(successful_paths)} images.")
            return successful_paths

    async def _download_single_image(
        self,
        session: aiohttp.ClientSession,
        image_result: dict,
        index: int,
        output_dir: Path,
        subject: str,
    ) -> Path | None:
        url = self._pick_best_image_url(image_result)
        if not url:
            self.logger.debug(f"Skipping result #{index+1}: no acceptable URL found.")
            return None

        for attempt in range(1, self.max_retries + 1):
            wait = self._rate_limit(url)
            if wait > 0:
                await self._sleep_with_jitter(wait)
            try:
                self.logger.debug(f"Attempt {attempt}/{self.max_retries}: Downloading {url}")
                async with session.get(url, timeout=self.timeout_s, allow_redirects=True) as response:
                    self._last_host_call[urlparse(url).netloc] = time.time()

                    if response.status == 429:
                        self.logger.warning(f"429 Too Many Requests for {url} (attempt {attempt}). Backing off and retrying once.")
                        await self._sleep_with_jitter(1.0 * attempt)
                        continue

                    if response.status != 200:
                        self.logger.warning(f"Failed to download {url} (status: {response.status}) on attempt {attempt}")
                        await self._sleep_with_jitter(0.75 * attempt)
                        continue

                    content_type = response.headers.get("Content-Type", "").lower()
                    if not content_type.startswith("image/"):
                        self.logger.warning(f"Skipping URL {url} as it is not a valid image (Content-Type: {content_type}).")
                        return None

                    content = await response.read()

                    try:
                        image_bytes = io.BytesIO(content)
                        with Image.open(image_bytes) as img:
                            converted_img = img.convert("RGB")
                            ext = ".jpg"
                            filename = f"{subject.replace(' ', '_')}_{index+1:03d}{ext}"
                            filepath = output_dir / filename
                            converted_img.save(filepath, "JPEG", quality=90)
                            self.logger.debug(f"Successfully downloaded and standardized {filepath}")
                            return filepath
                    except Exception as e:
                        self.logger.error(f"Failed to decode or save image from {url}. Error: {e}")
                        return None

            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout downloading {url} on attempt {attempt}")
            except Exception as e:
                self.logger.error(f"Unexpected error downloading {url}: {e}")
                break

            await self._sleep_with_jitter(0.75 * attempt)

        self.logger.error(f"Failed to download image from {url} after {self.max_retries} attempts.")
        return None

    def _filter_image_for_face(self, image_path: Path) -> bool:
        """
        Stricter human-face validation:
        - Requires at least one face bounding box.
        - Largest box must cover >= FACE_MIN_AREA_RATIO of the image area.
        - If landmarks API available, basic landmark presence check is applied.
        """
        if not FACE_REC_AVAILABLE:
            # Should not happen due to guard; keep file instead of deleting
            self.logger.warning("face_recognition unavailable during face filtering; keeping file.")
            return True
        try:
            image = face_recognition.load_image_file(image_path)  # type: ignore
            h, w = image.shape[:2]
            area = float(h * w) if h and w else 0.0
            if area <= 0:
                self.logger.warning(f"Image has invalid size: {image_path.name}. Deleting file.")
                image_path.unlink(missing_ok=True)
                return False

            face_locations = face_recognition.face_locations(image, model="hog")  # type: ignore
            if not face_locations:
                self.logger.warning(f"No human faces found in {image_path.name}. Deleting file.")
                image_path.unlink(missing_ok=True)
                return False

            # Check size threshold
            max_area_ratio = 0.0
            for top, right, bottom, left in face_locations:
                bw = max(0, right - left)
                bh = max(0, bottom - top)
                box_area = bw * bh
                if area > 0:
                    max_area_ratio = max(max_area_ratio, box_area / area)

            if max_area_ratio < self.FACE_MIN_AREA_RATIO:
                self.logger.warning(
                    f"Detected face too small ({max_area_ratio:.3%}) in {image_path.name}. Deleting file."
                )
                image_path.unlink(missing_ok=True)
                return False

            # Optional: landmarks check (if available)
            try:
                landmarks_list = face_recognition.face_landmarks(image)  # type: ignore
                if not landmarks_list or all(len(lm) == 0 for lm in landmarks_list):
                    self.logger.warning(f"No facial landmarks detected in {image_path.name}. Deleting file.")
                    image_path.unlink(missing_ok=True)
                    return False
            except Exception:
                # If landmarks fail, rely on bbox threshold only
                pass

            self.logger.info(f"✅ Valid human face(s) in {image_path.name}. Keeping file.")
            return True

        except Exception as e:
            self.logger.error(f"Error during face validation for {image_path.name}: {e}. Deleting file.")
            image_path.unlink(missing_ok=True)
            return False
