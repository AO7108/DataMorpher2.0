# Location: src/processors/class_builder.py

import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging
import inspect

from src.processors.bias_curator import curate_by_bias
from src.crawlers.google_images_crawler import GoogleImagesCrawler

logger = logging.getLogger(__name__)

def _count_images(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(1 for f in p.glob("*") if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))

async def build_classes(
    source_dir: str,
    out_root: str,
    subject: str,
    label_a: str,
    label_b: str,
    positive_prompt: str = "smiling",
    negative_prompt: str = "a person with a neutral expression, mouth closed, no teeth visible, not smiling",
    threshold: float = 0.25,
    probe_threshold: float = 0.20,
    enable_recrawl: bool = False,
    recrawl_prompts: List[str] | None = None,
    recrawl_cap: int = 150,
    crawler: GoogleImagesCrawler | None = None,
) -> Dict[str, Any]:
    """
    Modular class builder (not used directly by manager in the current wiring, but kept for reuse).
    """
    src = Path(source_dir)
    classes_dir = Path(out_root)
    classes_dir.mkdir(parents=True, exist_ok=True)

    class_a_dir = classes_dir / label_a
    class_b_dir = classes_dir / label_b
    class_a_dir.mkdir(parents=True, exist_ok=True)
    class_b_dir.mkdir(parents=True, exist_ok=True)

    # Primary curation (asymmetric thresholds)
    pos_th = float(threshold)
    neg_th = min(0.9, pos_th + 0.10)

    logger.info(f"[ClassBuilder] Primary curation into '{label_a}' and '{label_b}' (pos_th={pos_th}, neg_th={neg_th})")
    curate_by_bias(
        input_dir=str(src),
        output_dir=str(class_a_dir),
        prompt=positive_prompt,
        threshold=pos_th,
        neutral_prompt=negative_prompt,
    )
    curate_by_bias(
        input_dir=str(src),
        output_dir=str(class_b_dir),
        prompt=negative_prompt,
        threshold=neg_th,
        neutral_prompt=positive_prompt,
    )

    a_count = _count_images(class_a_dir)
    b_count = _count_images(class_b_dir)
    logger.info(f"[ClassBuilder] Counts after primary: {label_a}={a_count}, {label_b}={b_count}")

    # Probe if one empty
    if a_count == 0 or b_count == 0:
        probe_pos_th = max(0.10, pos_th - 0.05)
        probe_neg_th = min(0.95, neg_th + 0.05)
        tmp_a = classes_dir / f"_{label_a}_probe"
        tmp_b = classes_dir / f"_{label_b}_probe"
        for d in (tmp_a, tmp_b):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        curate_by_bias(
            input_dir=str(src),
            output_dir=str(tmp_a),
            prompt=positive_prompt,
            threshold=probe_pos_th,
            neutral_prompt=negative_prompt,
        )
        curate_by_bias(
            input_dir=str(src),
            output_dir=str(tmp_b),
            prompt=negative_prompt,
            threshold=probe_neg_th,
            neutral_prompt=positive_prompt,
        )

        if a_count == 0:
            for f in tmp_a.glob("*"):
                if f.is_file():
                    shutil.copy(f, class_a_dir / f.name)
        if b_count == 0:
            for f in tmp_b.glob("*"):
                if f.is_file():
                    shutil.copy(f, class_b_dir / f.name)

        for d in (tmp_a, tmp_b):
            try:
                shutil.rmtree(d)
            except Exception:
                pass

        a_count = _count_images(class_a_dir)
        b_count = _count_images(class_b_dir)
        logger.info(f"[ClassBuilder] Counts after probe: {label_a}={a_count}, {label_b}={b_count}")

    # Minority recrawl (optional)
    if enable_recrawl and crawler is not None:
        if a_count == 0 or b_count == 0 or abs(a_count - b_count) > max(5, min(a_count, b_count)):
            prompts = recrawl_prompts or [
                "not smiling",
                "neutral face",
                "serious expression",
                "closed mouth",
                "no teeth showing",
                "expressionless",
                "stoic face",
                "deadpan",
            ]
            minority_label = label_a if a_count < b_count else label_b
            queries = [f"{subject} {p}" for p in prompts][:6]
            per_query = max(3, recrawl_cap // max(1, len(queries)))

            recrawl_out = classes_dir / "_recrawl_raw"
            recrawl_out.mkdir(parents=True, exist_ok=True)
            downloaded = 0
            for rq in queries:
                try:
                    if inspect.iscoroutinefunction(crawler.scrape_with_query):
                        files = await crawler.scrape_with_query(rq, per_query, recrawl_out)
                    else:
                        files = crawler.scrape_with_query(rq, per_query, recrawl_out)  # type: ignore
                    downloaded += len(files or [])
                except Exception as e:
                    logger.warning(f"[ClassBuilder] Recrawl query failed '{rq}': {e}")

            if downloaded > 0:
                minority_dir = class_a_dir if minority_label == label_a else class_b_dir
                curate_by_bias(
                    input_dir=str(recrawl_out),
                    output_dir=str(minority_dir),
                    prompt=(negative_prompt if minority_label == label_b else positive_prompt),
                    threshold=max(0.15, pos_th - 0.05),
                    neutral_prompt=(positive_prompt if minority_label == label_b else negative_prompt),
                )
            else:
                logger.warning("[ClassBuilder] Recrawl did not fetch any new files.")

            try:
                shutil.rmtree(recrawl_out)
            except Exception:
                pass

            a_count = _count_images(class_a_dir)
            b_count = _count_images(class_b_dir)
            logger.info(f"[ClassBuilder] Counts after recrawl: {label_a}={a_count}, {label_b}={b_count}")

    return {
        "classes_dir": str(classes_dir),
        "class_a": {"label": label_a, "path": str(class_a_dir), "count": a_count},
        "class_b": {"label": label_b, "path": str(class_b_dir), "count": b_count},
    }
