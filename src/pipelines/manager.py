import inspect
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, AsyncGenerator
import shutil
import zipfile
from datetime import datetime
from tqdm import tqdm
import random

from src.parser.command_parser import CommandParser
from src.crawlers.google_images_crawler import GoogleImagesCrawler
from src.utils.models import DatasetMetadata
from src.processors.quality_scorer import quality_score
from src.processors.image_augmentor import augment_images
from src.processors.bias_curator import curate_by_bias
from src.processors.dataset_splitter import split_dataset
from src.processors.dataset_balancer import balance_dataset
from src.processors.quality_scorer import score_latest_split
from trainers.simple_classifier import train_classifier
from src.crawlers.reddit_crawler import RedditCrawler
from src.crawlers.youtube_crawler import YouTubeCrawler
from src.crawlers.twitter_crawler import TwitterCrawler
from src.crawlers.instagram_crawler import InstagramCrawler
from src.crawlers.tiktok_crawler import TikTokCrawler
from src.processors.face_classifier import get_expression
from src.utils.settings import load_settings
from ..config import config

def _band_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


class PipelineManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing PipelineManager...")
        self.parser = CommandParser()
        self.settings = load_settings()
        self.all_crawlers = {
            "google": GoogleImagesCrawler(),
            "youtube": YouTubeCrawler(),
            "reddit": RedditCrawler(),
            "tiktok": TikTokCrawler(),
            "twitter": TwitterCrawler(),
            "instagram": InstagramCrawler(),
        }
        self.crawlers = {
            name: crawler for name, crawler in self.all_crawlers.items()
            if not (hasattr(crawler, 'enabled') and not crawler.enabled)
        }
        self.logger.info(f"PipelineManager initialized with {len(self.crawlers)} active crawlers: {list(self.crawlers.keys())}")
        self.min_quality_band = str(self.settings.get("min_quality_band", "medium"))
        self.crawler_overrides: Dict[str, Any] = {}

    # --- DRY Helpers ---
    def _count_images(self, p: Path) -> int:
        if not p.exists():
            return 0
        return sum(
            1
            for f in p.glob("*")
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )

    def _retag_files(self, folder: Path, prefix: str) -> None:
        idx = 1
        for f in sorted(list(folder.glob("*"))):
            if not f.is_file():
                continue
            new_name = f"{prefix.replace(' ', '_')}_{idx:06d}{f.suffix.lower()}"
            target = folder / new_name
            if not target.exists() and f.name != new_name:
                try:
                    f.rename(target)
                    idx += 1
                except Exception:
                    pass

    def _downsample_to_match(self, folder: Path, target: int) -> None:
        files = [f for f in folder.glob("*") if f.is_file()]
        if len(files) > target:
            random.shuffle(files)
            for f in files[target:]:
                f.unlink(missing_ok=True)

    async def _recrawl_minority_if_needed(
        self,
        engine: str,
        labels: List[str],
        classes_dir: Path,
        class_a_dir: Path,
        class_b_dir: Path,
        a_count: int,
        b_count: int,
        active_sources: List[str],
        safe_subject: str,
        options: Dict[str, Any],
        pos_th: float | None = None,
        neutral_text: str | None = None,
    ) -> int:
        if not (a_count == 0 or b_count == 0 or abs(a_count - b_count) > max(5, min(a_count, b_count) * 0.3)):
            return 0
        minority_label = labels[0] if a_count < b_count else labels[1]
        minority_dir = class_a_dir if a_count < b_count else class_b_dir
        prompts = options.get("recrawl_minority_prompts") or ["not smiling", "neutral face"]
        recrawl_queries = [f"{safe_subject} {p}" for p in prompts][:6]
        cap = int(options.get("recrawl_cap", 150))
        per_query = max(3, cap // max(1, len(recrawl_queries)))
        crawler = self.crawlers.get(active_sources[0] if active_sources else None)
        if not crawler:
            return 0
        recrawl_out = classes_dir / "_recrawl_raw"
        recrawl_out.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        for rq in recrawl_queries:
            try:
                files = await crawler.scrape_with_query(rq, per_query, recrawl_out)
                downloaded += len(files or [])
            except Exception:
                pass
        added_count = 0
        if downloaded > 0:
            if engine == "DeepFace (Faces Only)":
                minority_expression = 'smiling' if a_count < b_count else 'non smiling'
                for image_path in tqdm(recrawl_out.glob("*"), desc="Curating Recrawled"):
                    if not image_path.is_file():
                        continue
                    try:
                        expression = get_expression(str(image_path))
                        if expression == minority_expression:
                            shutil.copy(image_path, minority_dir / image_path.name)
                            added_count += 1
                    except Exception:
                        pass
            else:
                # CLIP Recrawl Curation
                curate_by_bias(
                    input_dir=str(recrawl_out),
                    output_dir=str(minority_dir),
                    prompt=(neutral_text if minority_label == labels[1] else labels[0]),
                    threshold=max(0.15, (pos_th or 0.25) - 0.05),
                    neutral_prompt=(labels[0] if minority_label == labels[1] else (neutral_text or "")),
                    strict_negative=(minority_label == labels[1]),
                )
        shutil.rmtree(recrawl_out, ignore_errors=True)
        return added_count

    def _cleanup_and_balance(
        self,
        engine: str,
        labels: List[str],
        classes_dir: Path,
        class_a_dir: Path,
        class_b_dir: Path,
        pos_th: float | None,
        neutral_text: str | None,
    ) -> tuple[int, int]:
        if engine == "DeepFace (Faces Only)":
            for image_path in tqdm(list(class_b_dir.glob("*")), desc=f"Cleaning '{labels[1]}'"):
                if image_path.is_file() and get_expression(str(image_path)) == 'smiling':
                    image_path.unlink()
            for image_path in tqdm(list(class_a_dir.glob("*")), desc=f"Cleaning '{labels[0]}'"):
                if image_path.is_file() and get_expression(str(image_path)) == 'non smiling':
                    image_path.unlink()
        else:
            tmp_smile_in_neg = classes_dir / "_smile_in_neg"
            if tmp_smile_in_neg.exists():
                shutil.rmtree(tmp_smile_in_neg)
            tmp_smile_in_neg.mkdir(parents=True, exist_ok=True)
            curate_by_bias(
                input_dir=str(class_b_dir),
                output_dir=str(tmp_smile_in_neg),
                prompt=labels[0],
                threshold=(pos_th or 0.25),
                neutral_prompt=(neutral_text or ""),
            )
            for f in tmp_smile_in_neg.glob("*"):
                try:
                    (class_b_dir / f.name).unlink(missing_ok=True)
                except Exception:
                    pass
            shutil.rmtree(tmp_smile_in_neg, ignore_errors=True)

        a_count, b_count = self._count_images(class_a_dir), self._count_images(class_b_dir)
        minority = min(a_count, b_count)
        if minority > 0 and a_count != b_count:
            self._downsample_to_match(class_a_dir, minority)
            self._downsample_to_match(class_b_dir, minority)
            a_count, b_count = self._count_images(class_a_dir), self._count_images(class_b_dir)
        return a_count, b_count

    async def dispatch(
        self, user_request: str, sources: Dict[str, bool], options: Dict[str, Any]
    ) -> AsyncGenerator[str | DatasetMetadata, None]:
        yield f"--- Dispatching new user request: '{user_request}' ---"

        command = self.parser.parse(user_request)
        if not command or command.get("intent") != "create_dataset":
            yield "[ERROR] Failed to parse a valid 'create_dataset' command. Aborting."
            return

        target_count: int = command.get("count", 18) 
        subject: str | None = command.get("subject") 
        attributes: List[str] = command.get("attributes", [])

       
        if target_count is None or target_count <= 0:
            yield "[INFO] Target count is zero or invalid. Nothing to create."
            return
       

        if not subject or not isinstance(subject, str) or not subject.strip():
            yield "[ERROR] Could not determine subject from the prompt."
            return

        safe_subject = subject.strip()
        safe_subject_slug = safe_subject.replace(" ", "_")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_slug = f"{safe_subject_slug}_{timestamp}"
        base_outputs = Path(config["paths"]["outputs"])
        run_dir = base_outputs / version_slug
        run_dir.mkdir(parents=True, exist_ok=True)
        yield f"📂 Versioned output directory created: {run_dir}"

        parsed_class_split: bool = bool(command.get("class_split"))
        parsed_class_labels: List[str] | None = command.get("class_labels")
        active_sources = [source for source, is_active in sources.items() if is_active]

        want_classes_explicit = bool(options.get("organize_curated_as_classes", False))
        is_two_class_run = want_classes_explicit or parsed_class_split
        collected_clean_files: Set[Path] = set()

        # --- SMART CRAWLING LOGIC ---
        if is_two_class_run:
            yield "[INFO] Two-class workflow detected. Performing targeted crawl for each class."
            labels = options.get("class_labels") or parsed_class_labels or ["class_a", "class_b"]
            count_per_class = target_count // 2

            for label in labels:
                query_attrs = [attr for attr in attributes if attr not in labels] + [label]
                yield f"--- Starting targeted crawl for class: '{label}' ---"
                
                num_to_fetch = int(count_per_class * 1.8) + 10
                output_dir = Path(f"data/raw/{safe_subject_slug}/{label.replace(' ', '_')}")
                crawler = self.crawlers.get(active_sources[0] if active_sources else None)

                if crawler:
                    try:
                        metadata_batch = await crawler.scrape(subject, num_to_fetch, query_attrs, output_dir)
                        if metadata_batch and metadata_batch.files:
                            collected_clean_files.update(set(metadata_batch.files))
                    except Exception as e:
                         yield f"[CRITICAL ERROR] Crawl for class '{label}' failed: {e}"
        else:
            yield "[INFO] Single-concept workflow detected. Performing standard crawl."
            current_attempt = 1
            max_attempts = 5
            while len(collected_clean_files) < target_count and current_attempt <= max_attempts:
                needed_files = target_count - len(collected_clean_files)
                yield (f"\n[Attempt {current_attempt}/{max_attempts}] Goal: {target_count}. "
                       f"Collected: {len(collected_clean_files)}. Need {needed_files} more.")
                
                crawler = self.crawlers.get(active_sources[0] if active_sources else None)
                if not crawler:
                    yield f"[ERROR] No active and registered crawler found."
                    break
                
                num_to_fetch = int(needed_files * 1.5) + 5
                output_dir = Path(f"data/raw/{safe_subject_slug}")
                try:
                    metadata_batch = await crawler.scrape(subject, num_to_fetch, attributes, output_dir)
                    new_files_this_attempt = 0
                    if metadata_batch and metadata_batch.files:
                        prev_count = len(collected_clean_files)
                        collected_clean_files.update(set(metadata_batch.files))
                        new_files_this_attempt = len(collected_clean_files) - prev_count
                    
                    yield f"Crawler task finished. Added {new_files_this_attempt} new unique files."
                    if new_files_this_attempt == 0 and current_attempt > 1:
                        yield "[WARNING] No new files found. Stopping early."
                        break
                    current_attempt += 1
                except Exception as e:
                    yield f"[CRITICAL ERROR] An error occurred during crawl attempt: {e}"
                    break

        yield f"\n--- Loop finished. Collected {len(collected_clean_files)} files (pre-quality). ---"

        # ===== Quality-backed selection to guarantee minimum kept files =====
        def score_and_filter(files: List[Path]) -> List[Path]:
            order = {"low": 0, "medium": 1, "high": 2}
            return [p for p in files if order.get(_band_from_score(quality_score(p)), 0) >= order[self.min_quality_band]]

        kept_files = score_and_filter(list(collected_clean_files))
        yield f"Quality filtering complete. Kept {len(kept_files)} of {len(collected_clean_files)} images."

        max_extra_quality_passes = 2
        extra_pass = 1
        while len(kept_files) < target_count and extra_pass <= max_extra_quality_passes:
            deficit = target_count - len(kept_files)
            ask = int(deficit * 1.5) + 5
            yield f"[QUALITY] Kept {len(kept_files)}/{target_count}. Extra crawl ~{ask} files (pass {extra_pass}/{max_extra_quality_passes})."

            crawler = self.crawlers.get(active_sources[0] if active_sources else None)
            if not crawler:
                yield "[QUALITY] No crawler available. Stopping quality-backed loop."
                break

            try:
                output_dir = Path(f"data/raw/{safe_subject_slug}")
                meta_more: DatasetMetadata | None = None
                if inspect.iscoroutinefunction(crawler.scrape):
                    meta_more = await crawler.scrape(safe_subject, ask, attributes, output_dir)
                else:
                    meta_more = crawler.scrape(safe_subject, ask, attributes, output_dir)

                if meta_more and meta_more.files:
                    prev = len(collected_clean_files)
                    collected_clean_files.update(set(meta_more.files))
                    added = len(collected_clean_files) - prev
                    yield f"[QUALITY] Extra crawl added {added} new unique files."
                    kept_files = score_and_filter(list(collected_clean_files))
                else:
                    yield "[QUALITY] Extra crawl returned no new files."
            except Exception as e:
                yield f"[QUALITY] Extra crawl failed: {e}"
                break
            extra_pass += 1

        if not kept_files:
            yield "[FINAL WARNING] Crawl and quality filter resulted in zero images. Aborting."
            return

        work_dir = run_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        for p in kept_files:
            try:
                shutil.copy(p, work_dir / p.name, follow_symlinks=False)
            except (FileNotFoundError, shutil.SameFileError):
                pass
        last_stage_dir = work_dir

        try:
            if options.get("augment") and not is_two_class_run:
                out_dir = run_dir / "augmented"
                yield f"Running augmentation for single-concept dataset → {out_dir}"
                augment_images(str(last_stage_dir), str(out_dir))
                last_stage_dir = out_dir

            if is_two_class_run:
                # Use DRY helpers
                
                curation_engine = options.get("curation_engine", "CLIP (General Purpose)")
                labels = options.get("class_labels") or parsed_class_labels or ["smiling", "non smiling"]
                classes_dir = run_dir / "classes"
                class_a_dir, class_b_dir = classes_dir / labels[0], classes_dir / labels[1]
                class_a_dir.mkdir(parents=True, exist_ok=True); class_b_dir.mkdir(parents=True, exist_ok=True)

                if curation_engine == "DeepFace (Faces Only)":
                    yield "[ClassBuilder] Using 'DeepFace' engine."
                    for image_path in tqdm(list(last_stage_dir.glob("*")), desc="Initial Classification"):
                        if not image_path.is_file(): continue
                        expression = get_expression(str(image_path))
                        if expression == 'smiling': shutil.copy(image_path, class_a_dir / image_path.name)
                        elif expression == 'non smiling': shutil.copy(image_path, class_b_dir / image_path.name)
                else: # CLIP Workflow
                    yield "[ClassBuilder] Using 'CLIP (General Purpose)' engine."
                    pos_th = float(options.get("curate_threshold", 0.25))
                    neg_th = min(0.95, pos_th + 0.15)
                    neutral_text = options.get("curate_neutral", "a person with a neutral expression")
                    neutral_text += ", neutral gaze, relaxed lips, no grin, no visible teeth"
                    
                    yield f"Curating into '{labels[0]}'"
                    curate_by_bias(input_dir=str(last_stage_dir), output_dir=str(class_a_dir), prompt=labels[0], threshold=pos_th, neutral_prompt=neutral_text)
                    yield f"Curating into '{labels[1]}'"
                    curate_by_bias(input_dir=str(last_stage_dir), output_dir=str(class_b_dir), prompt=neutral_text, threshold=neg_th, neutral_prompt=labels[0], strict_negative=True)

                a_count, b_count = self._count_images(class_a_dir), self._count_images(class_b_dir)
                yield f"[ClassBuilder] Initial counts: {labels[0]}={a_count}, {labels[1]}={b_count}"

                # Invoke minority recrawl helper (no internal yields). Log around the call.
                yield "[ClassBuilder] Checking for imbalance to trigger recrawl if needed."
                recrawl_opts = dict(options)
                added = await self._recrawl_minority_if_needed(
                    engine=curation_engine,
                    labels=labels,
                    classes_dir=classes_dir,
                    class_a_dir=class_a_dir,
                    class_b_dir=class_b_dir,
                    a_count=a_count,
                    b_count=b_count,
                    active_sources=active_sources,
                    safe_subject=safe_subject,
                    options=recrawl_opts,
                    pos_th=pos_th if 'pos_th' in locals() else None,
                    neutral_text=neutral_text if 'neutral_text' in locals() else None,
                )
                if added:
                    yield f"[ClassBuilder] Added {added} new images to minority class."

                yield "[ClassBuilder] Starting final two-way cleanup pass..."
                if curation_engine == "DeepFace (Faces Only)":
                    for image_path in tqdm(list(class_b_dir.glob("*")), desc=f"Cleaning '{labels[1]}'"):
                        if image_path.is_file() and get_expression(str(image_path)) == 'smiling': image_path.unlink()
                    for image_path in tqdm(list(class_a_dir.glob("*")), desc=f"Cleaning '{labels[0]}'"):
                        if image_path.is_file() and get_expression(str(image_path)) == 'non smiling': image_path.unlink()
                else: # CLIP Cleanup
                    # Smile-exclusion cleanup from non smiling folder
                    tmp_smile_in_neg = classes_dir / "_smile_in_neg"
                    if tmp_smile_in_neg.exists():
                        shutil.rmtree(tmp_smile_in_neg)
                    tmp_smile_in_neg.mkdir(parents=True, exist_ok=True)

                    curate_by_bias(
                        input_dir=str(class_b_dir),
                        output_dir=str(tmp_smile_in_neg),
                        prompt=labels[0],
                        threshold=pos_th,  # can use pos_th+0.05 for stricter exclusion
                        neutral_prompt=neutral_text,
                    )
                    removed = 0
                    for f in tmp_smile_in_neg.glob("*"):
                        try:
                            (class_b_dir / f.name).unlink(missing_ok=True)
                            removed += 1
                        except Exception:
                            pass
                    try:
                        shutil.rmtree(tmp_smile_in_neg)
                    except Exception:
                        pass
                    yield f"[ClassCleaner] Removed {removed} smiling-leaning files from '{labels[1]}'"


                a_count, b_count = self._cleanup_and_balance(
                    engine=curation_engine,
                    labels=labels,
                    classes_dir=classes_dir,
                    class_a_dir=class_a_dir,
                    class_b_dir=class_b_dir,
                    pos_th=pos_th if 'pos_th' in locals() else None,
                    neutral_text=neutral_text if 'neutral_text' in locals() else None,
                )

                if options.get("augment"):
                    yield "[INFO] Augmenting classified folders..."
                    for cdir, label in zip([class_a_dir, class_b_dir], labels):
                        if cdir.exists() and any(cdir.iterdir()):
                            yield f"Augmenting class: {label}"
                            augment_images(str(cdir), str(cdir))
                
                self._retag_files(class_a_dir, labels[0])
                self._retag_files(class_b_dir, labels[1])
                last_stage_dir = classes_dir

            if options.get("split"):
                out_dir = run_dir / "split"
                yield f"Splitting dataset → {out_dir}"
                split_dataset(str(last_stage_dir), str(out_dir))
                last_stage_dir = out_dir
            
            if options.get("score"):
                report_path = run_dir / "quality_report.json"
                yield f"Scoring quality → {report_path}"
                score_latest_split(base_dir=str(run_dir), output_file=str(report_path))
            
            if options.get("augment_after_split"):
                yield "Augmenting each split subset in place."
                for subset in ("train", "val", "test"):
                    subset_dir = Path(last_stage_dir) / subset
                    if subset_dir.exists():
                        class_dirs = [d for d in subset_dir.iterdir() if d.is_dir()]
                        if class_dirs:
                            for cdir in class_dirs:
                                try:
                                    augment_images(str(cdir), str(cdir))
                                except Exception as e:
                                    yield f"[AUGMENT POST-SPLIT] Failed in {cdir}: {e}"
                        else:
                            try:
                                augment_images(str(subset_dir), str(subset_dir))
                            except Exception as e:
                                yield f"[AUGMENT POST-SPLIT] Failed in {subset_dir}: {e}"
            
            if options.get("train"):
                split_dir = run_dir / "split"
                if split_dir.exists() and any(split_dir.iterdir()):
                    models_dir = run_dir / "models"
                    cfg = options.get("train_cfg", {}) or {}
                    
                    cfg["is_simple_dataset"] = not is_two_class_run

                    yield f"Training classifier on split dataset → {models_dir}"
                    try:
                        result = train_classifier(
                            data_dir=str(split_dir),
                            out_dir=str(models_dir),
                            config=cfg,
                        )
                        if not result.get("skipped"):
                            model_path = result.get("model_path")
                            metrics_path = result.get("metrics_path")
                            yield (f"Trainer finished. Model: {model_path}, Metrics: {metrics_path}, "
                                   f"Best Val Acc: {result.get('best_val_acc', 0):.3f}")
                        else:
                            yield f"[INFO] Training skipped: {result.get('reason')}"
                    except Exception as e:
                        yield f"[WARNING] Training failed: {e}"
                else:
                    yield "[INFO] Training skipped: split directory not found or empty."

        except Exception as e:
            yield f"[WARNING] Post-processing pipeline encountered an error: {e}"

        # --- FINAL STEPS ---
        zip_path = run_dir / "dataset.zip"
        try:
            yield f"Packaging final dataset → {zip_path}"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in last_stage_dir.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=f.relative_to(last_stage_dir))
        except Exception as e:
            yield f"[WARNING] Failed to create ZIP: {e}"
        
        model_path = run_dir / "models"
        metrics_path = run_dir / "models"
        yield (f"ARTIFACTS:\n"
               f"- Final stage dir: {last_stage_dir}\n"
               f"- Quality report (if scored): {run_dir / 'quality_report.json'}\n"
               f"- Model (if trained): {model_path if options.get('train') else 'n/a'}\n"
               f"- Metrics (if trained): {metrics_path if options.get('train') else 'n/a'}\n"
               f"- Download ZIP: {zip_path}")

        yield f"--- ✅ PIPELINE FINISHED SUCCESSFULLY! ---"
        
        final_files = list(p for p in last_stage_dir.rglob("*") if p.is_file())
        yield DatasetMetadata(
            subject=safe_subject,
            modality="image",
            source=", ".join(active_sources),
            attributes=attributes,
            files=final_files
        )
       