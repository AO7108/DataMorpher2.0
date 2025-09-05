import cv2
from pathlib import Path
import json
from tqdm import tqdm
import logging
import os
from collections import defaultdict
from typing import List

# --- Dependency Imports for Aesthetic Model ---
# This will be lazy-loaded to avoid loading heavy libraries if not needed.
try:
    import torch
    import pyiqa
    from PIL import Image
    HAS_AESTHETICS = True
except ImportError:
    HAS_AESTHETICS = False

# --- Global variables for lazy loading the aesthetic model ---
aesthetic_model = None
device = None

# --- Logging Setup ---
logging.basicConfig(format="%(asctime)s - [%(levelname)s] - %(message)s", level=logging.INFO)


def _initialize_aesthetic_model():
    """Initializes the aesthetic model only once when first needed."""
    global aesthetic_model, device
    
    if not HAS_AESTHETICS:
        logging.warning("Aesthetic scoring is disabled. Please install 'pyiqa', 'torch', and 'timm'.")
        return False
        
    if aesthetic_model is None:
        try:
            logging.info("Initializing aesthetic scoring model (this may download weights on first run)...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # We use the NIMA model, known for predicting aesthetic quality on a 1-10 scale.
            aesthetic_model = pyiqa.create_metric('nima', device=device)
            logging.info(f"✅ Aesthetic model loaded successfully on device: {device}")
        except Exception as e:
            logging.error(f"Failed to initialize aesthetic model: {e}", exc_info=True)
            return False
            
    return True

def _calculate_blur_score(image_path: Path) -> float:
    """Calculate Laplacian variance for sharpness."""
    try:
        image = cv2.imread(str(image_path))
        if image is None: return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0

_MIN_VAR = 50.0
_MAX_VAR = 1000.0

def quality_score(image_path: Path) -> float:
    """Returns a normalized quality score in [0.0, 1.0] based on blur variance."""
    var = _calculate_blur_score(image_path)
    v = max(_MIN_VAR, min(_MAX_VAR, var))
    return round((v - _MIN_VAR) / (_MAX_VAR - _MIN_VAR), 4)

def quality_scores_batch(paths: List[Path]) -> List[float]:
    """Convenience helper to score many paths quickly."""
    return [quality_score(p) for p in paths]

def _calculate_aesthetic_score(image_path: Path) -> float:
    """
    Calculate an aesthetic score for an image using the NIMA model.
    Returns a score on a scale of ~1 to 10.
    """
    if aesthetic_model is None:
        if not _initialize_aesthetic_model():
            return 0.0 # Return a neutral score if model fails to load

    try:
        # The model expects a torch tensor
        score = aesthetic_model(str(image_path)).item()
        return round(score, 2)
    except Exception as e:
        logging.warning(f"Could not calculate aesthetic score for {image_path.name}: {e}")
        return 0.0

def get_unified_quality_score(image_path: Path) -> dict:
    """
    Calculates a unified quality report for a single image, including blur,
    aesthetic, and a final combined score.
    """
    blur_score_raw = _calculate_blur_score(image_path)
    normalized_blur = quality_score(image_path)
    
    aesthetic_score = _calculate_aesthetic_score(image_path)
    # Normalize aesthetic score (NIMA is ~1-10, so we map it to 0-1)
    normalized_aesthetic = max(0.0, min(1.0, (aesthetic_score - 1.0) / 9.0))
    
    # Combine scores with weighting. Sharpness (blur) is often more critical.
    unified_score = (normalized_blur * 0.7) + (normalized_aesthetic * 0.3)
    
    return {
        "blur_score": round(blur_score_raw, 2),
        "aesthetic_score": aesthetic_score,
        "unified_quality_score": round(unified_score, 4)
    }

def _score_directory(input_dir: str, output_file: str):
    """Core worker function to analyze a specific directory for quality."""
    # Initialize the model once before the loop if aesthetics will be calculated
    _initialize_aesthetic_model()

    input_path = Path(input_dir)
    image_files = [f for f in input_path.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    if not image_files:
        logging.warning(f"⚠️ No images found in {input_path}")
        return

    all_image_details = {}
    total_blur_score = 0
    total_unified_score = 0
    class_scores = defaultdict(list)
    class_unified_scores = defaultdict(list)

    logging.info(f"📊 Scoring quality of {len(image_files)} images from '{input_path.name}'...")
    for image_path in tqdm(image_files, desc="Scoring Images"):
        quality_metrics = get_unified_quality_score(image_path)
        
        total_blur_score += quality_metrics["blur_score"]
        total_unified_score += quality_metrics["unified_quality_score"]
        
        class_name = image_path.parent.name
        class_scores[class_name].append(quality_metrics["blur_score"])
        class_unified_scores[class_name].append(quality_metrics["unified_quality_score"])

        all_image_details[str(image_path.relative_to(input_path))] = quality_metrics

    overall_average_blur_score = total_blur_score / len(image_files) if image_files else 0
    overall_average_unified_score = total_unified_score / len(image_files) if image_files else 0

    class_summary_blur = {cls: round(sum(scores) / len(scores), 2) for cls, scores in class_scores.items()}
    class_summary_unified = {cls: round(sum(scores) / len(scores), 4) for cls, scores in class_unified_scores.items()}
    
    final_report = {
        "summary": {
            "scored_dataset": str(input_path),
            "total_images_scored": len(image_files),
            "overall_average_blur_score": round(overall_average_blur_score, 2),
            "overall_average_unified_score": round(overall_average_unified_score, 4),
            "per_class_average_blur_score": class_summary_blur,
            "per_class_average_unified_score": class_summary_unified
        },
        "image_details": all_image_details
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_report, f, indent=4)
    logging.info(f"✅ Quality report saved: {output_path}")

def score_latest_split(base_dir: str, output_file: str):
    """
    Finds the latest dataset split (train/val/test) inside base_dir and scores it.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        logging.error(f"❌ Base directory not found: {base_path}")
        return

    # Logic to find the latest directory to score
    # This assumes the target is inside a timestamped version folder
    subdirectories = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirectories:
        logging.error(f"❌ No dataset folders found inside {base_path}")
        return
        
    latest_dir = max(subdirectories, key=lambda p: p.stat().st_mtime)
    logging.info(f"🎯 Found latest dataset to score: '{latest_dir.name}'")

    _score_directory(input_dir=str(latest_dir), output_file=output_file)