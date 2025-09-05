import shutil
import logging
from pathlib import Path
from typing import List, Optional

from src.utils.models import DatasetMetadata

# Dependency checks
try:
    import yt_dlp  # type: ignore
    HAS_YTDLP = True
except ImportError:
    HAS_YTDLP = False

try:
    import ffmpeg  # type: ignore
    HAS_FFMPEG = True
except ImportError:
    HAS_FFMPEG = False


class YouTubeCrawler:
    """
    A crawler for fetching video or audio clips from YouTube using yt-dlp.
    - If 'audio' is specified in attributes, it extracts audio tracks using ffmpeg.
    - The crawler is disabled if yt-dlp is not installed.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if not HAS_YTDLP:
            self.logger.warning("YouTubeCrawler is disabled: 'yt-dlp' library not found. Please install it.")
            self.enabled = False
            return

        self.enabled = True
        self.logger.info("YouTubeCrawler initialized successfully.")
        if not HAS_FFMPEG:
            self.logger.warning("ffmpeg-python not found. Audio extraction will be disabled.")

    async def scrape(self, subject: str, limit: int, attributes: List[str], output_dir: Path) -> Optional[DatasetMetadata]:
        if not self.enabled:
            return None

        query = self._build_query(subject, attributes)
        video_dir = output_dir / "videos_raw"
        
        self.logger.info(f"Searching YouTube with query: '{query}'")
        downloaded_videos = self._download_videos(query, limit, video_dir)

        if not downloaded_videos:
            self.logger.warning("No videos were downloaded from YouTube.")
            return None

        # Check if the user specifically requested audio
        wants_audio = any(attr.lower() == "audio" for attr in attributes)

        if wants_audio:
            if not HAS_FFMPEG:
                self.logger.error("Cannot extract audio: ffmpeg-python is not installed. Returning video files instead.")
                modality = "video"
                final_files = downloaded_videos
            else:
                self.logger.info("Audio extraction requested. Processing video files...")
                audio_dir = output_dir / "audio"
                extracted_audio = self._extract_audio(downloaded_videos, audio_dir)
                if not extracted_audio:
                    self.logger.warning("Audio extraction failed for all videos. No dataset produced.")
                    return None
                modality = "audio"
                final_files = extracted_audio
        else:
            modality = "video"
            final_files = downloaded_videos
            # If only videos are needed, we can move them to the parent dir
            for f in final_files:
                try:
                    shutil.move(str(f), str(output_dir / f.name))
                except Exception:
                    pass


        self.logger.info(f"YouTube crawl successful. Created a dataset of {len(final_files)} {modality} files.")
        return DatasetMetadata(
            subject=subject,
            modality=modality,
            source="youtube",
            attributes=attributes or [],
            files=final_files
        )

    async def scrape_with_query(self, query: str, limit: int, output_dir: Path) -> List[Path]:
        if not self.enabled:
            return []
        
        # For recrawl, we assume video is the target unless specified
        video_dir = output_dir / "videos_raw"
        downloaded_videos = self._download_videos(query, limit, video_dir)
        return downloaded_videos or []

    def _build_query(self, subject: str, attributes: List[str]) -> str:
        parts = [subject] + (attributes or [])
        # Remove 'audio' from query as it's a processing instruction
        query_parts = [p for p in parts if p.lower() != 'audio']
        return " ".join(query_parts).strip()

    def _download_videos(self, query: str, limit: int, out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # yt-dlp options: download best mp4, quiet logging, ignore errors on individual videos
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(out_dir / "%(title).80s-%(id)s.%(ext)s"),
            "noplaylist": True,
            "ignoreerrors": True,
            "quiet": True,
            "no_warnings": True,
            "max_downloads": limit,
            "default_search": f"ytsearch{limit}",
        }

        try:
            self.logger.info(f"Downloading up to {limit} videos...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([query])
            
            # Find all downloaded files that are not temporary
            downloaded_files = [p for p in out_dir.iterdir() if p.is_file() and not p.name.endswith((".part", ".ytdl"))]
            self.logger.info(f"Successfully downloaded {len(downloaded_files)} videos.")
            return downloaded_files

        except Exception as e:
            self.logger.error(f"An unexpected error occurred during YouTube download: {e}", exc_info=True)
            return []

    def _extract_audio(self, video_paths: List[Path], out_dir: Path) -> List[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        extracted_files: List[Path] = []
        
        self.logger.info(f"Extracting audio from {len(video_paths)} videos...")
        for video_path in video_paths:
            output_path = out_dir / (video_path.stem + ".mp3")
            try:
                (
                    ffmpeg
                    .input(str(video_path))
                    .output(str(output_path), acodec="libmp3lame", audio_bitrate="192k", ar=44100, loglevel="error")
                    .overwrite_output()
                    .run()
                )
                if output_path.exists():
                    extracted_files.append(output_path)
            except Exception as e:
                self.logger.warning(f"Could not extract audio from {video_path.name}: {e}")
                continue
        
        self.logger.info(f"Successfully extracted audio from {len(extracted_files)} files.")
        return extracted_files