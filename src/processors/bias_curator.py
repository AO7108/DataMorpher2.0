# Location: src/processors/bias_curator.py

import logging
from pathlib import Path
import shutil
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# Optional transformers/torch; handle gracefully
try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_TRANSFORMERS = True
except Exception:
    torch = None  # type: ignore
    CLIPProcessor = None  # type: ignore
    CLIPModel = None  # type: ignore
    HAS_TRANSFORMERS = False

# --- Global variables for lazy loading ---
clip_model = None
clip_processor = None
device = "cuda" if (HAS_TRANSFORMERS and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def _initialize_clip() -> bool:
    """Initializes the CLIP model and processor if available."""
    global clip_model, clip_processor
    if not HAS_TRANSFORMERS:
        logging.warning("transformers not installed; skipping CLIP curation.")
        return False

    if clip_model is None or clip_processor is None:
        try:
            logging.info("🤖 Loading CLIP model (this may take a moment on first run)...")
            model_id = "openai/clip-vit-base-patch32"
            clip_model = CLIPModel.from_pretrained(model_id).to(device)
            clip_processor = CLIPProcessor.from_pretrained(model_id)
            logging.info(f"✅ CLIP model loaded on device: {device}")
        except Exception as e:
            logging.warning(f"Failed to load CLIP model ({e}); skipping curation.")
            return False

    return True


def curate_by_bias(
    input_dir: str,
    output_dir: str,
    prompt: str,
    threshold: float = 0.25,
    neutral_prompt: str = "a generic photo",
    strict_negative: bool = False,
):
    """
    Filters a dataset of images based on similarity to a given prompt.
    Gracefully no-ops if CLIP is unavailable.

    strict_negative: when True, strengthens the neutral_prompt slightly to reduce false positives.
    """
    if not _initialize_clip():
        logging.warning("CLIP unavailable; curation step skipped.")
        return

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logging.error(f"❌ Input directory not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"📂 Output directory created at: {output_path}")

    image_files = [f for f in input_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    if not image_files:
        logging.warning(f"⚠️ No images found in {input_path}")
        return

    logging.info(f"🔍 Curating {len(image_files)} images with prompt: '{prompt}' (threshold={threshold})")

    # Slightly stronger neutral text if requested; default remains unchanged
    neutral_text = neutral_prompt
    if strict_negative and neutral_prompt:
        neutral_text = neutral_prompt + ", neutral gaze, relaxed lips, no grin, no visible teeth"

    kept_count = 0
    batch_size = 8

    for i in tqdm(range(0, len(image_files), batch_size), desc="Curating images"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        for image_path in batch_files:
            try:
                img = Image.open(image_path).convert("RGB")
                batch_images.append((image_path, img))
            except UnidentifiedImageError:
                logging.warning(f"❗️ Skipping corrupted or unsupported file: {image_path.name}")
        if not batch_images:
            continue

        try:
            inputs = clip_processor(
                text=[prompt, neutral_text],
                images=[img for _, img in batch_images],
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():  # type: ignore
                outputs = clip_model(**inputs)  # type: ignore
                probs = outputs.logits_per_image.softmax(dim=1)  # type: ignore
        except Exception as e:
            logging.warning(f"CLIP inference failed: {e}. Skipping remaining curation.")
            return

        for (image_path, _), prob in zip(batch_images, probs):
            score = float(prob[0].item())
            if score >= threshold:
                try:
                    shutil.copy(image_path, output_path / image_path.name)
                    kept_count += 1
                except Exception as e:
                    logging.warning(f"Failed to copy curated file {image_path.name}: {e}")

    logging.info(
        f"✅ Curation complete. Kept {kept_count} of {len(image_files)} images "
        f"({(kept_count/len(image_files))*100:.2f}% match rate)."
    )
