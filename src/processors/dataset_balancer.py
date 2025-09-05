import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
import albumentations as A
import cv2
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import logging
import os

# --- Logging Setup ---
logging.basicConfig(format="%(asctime)s - [%(levelname)s] - %(message)s", level=logging.INFO)

# --- Global variables ---
clip_model = None
clip_processor = None
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

def _initialize_clip():
    """Initializes CLIP only once."""
    global clip_model, clip_processor
    if clip_model is None:
        logging.info("🤖 Loading CLIP model for classification...")
        model_id = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(model_id).to(DEVICE_STR)
        clip_processor = CLIPProcessor.from_pretrained(model_id)
        logging.info(f"✅ CLIP model loaded on: {DEVICE_STR}")

def _safe_open_image(image_path: Path):
    """Opens an image safely and converts it to RGB."""
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB")
    except UnidentifiedImageError:
        logging.warning(f"⚠️ Unreadable image: {image_path.name}")
    except Exception as e:
        logging.warning(f"❗️ Error opening {image_path.name}: {e}")
    return None

def balance_dataset(
    input_dir: str,
    output_dir: str,
    positive_prompt: str,
    negative_prompt: str = "a photo of a person with a neutral expression",
    mode: str = 'downsample',
    overwrite: bool = True
):
    """
    Balances a dataset using CLIP classification.

    Args:
        mode: 'downsample' reduces majority, 'upsample' augments minority.
        overwrite: If True, deletes existing output_dir.
    """
    _initialize_clip()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logging.error(f"❌ Input directory not found: {input_path}")
        return

    # Handle output dir
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
            logging.info(f"🗑️ Cleared existing folder: {output_path}")
        else:
            logging.info(f"📂 Appending to existing folder: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = [f for f in input_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    if not image_files:
        logging.warning(f"⚠️ No images found in {input_path}")
        return

    # --- Classification ---
    logging.info(f"🔍 Classifying {len(image_files)} images...")
    positive_files, negative_files = [], []

    for image_path in tqdm(image_files, desc="Classifying Images"):
        image = _safe_open_image(image_path)
        if image is None:
            continue

        inputs = clip_processor(
            text=[positive_prompt, negative_prompt],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(DEVICE_STR)

        with torch.no_grad():
            outputs = clip_model(**inputs)

        if outputs.logits_per_image[0][0] > outputs.logits_per_image[0][1]:
            positive_files.append(image_path)
        else:
            negative_files.append(image_path)

    logging.info(f"📊 Results: {len(positive_files)} positive / {len(negative_files)} negative")

    # --- Balancing ---
    if len(positive_files) == len(negative_files):
        logging.info("✅ Already balanced. Copying all files.")
        final_files = positive_files + negative_files

    elif mode == 'downsample':
        majority, minority = (positive_files, negative_files) if len(positive_files) > len(negative_files) else (negative_files, positive_files)
        random.shuffle(majority)
        downsampled = majority[:len(minority)]
        final_files = downsampled + minority
        logging.info(f"⚖️ Downsampled from {len(majority)} to {len(downsampled)}.")

    elif mode == 'upsample':
        majority, minority = (positive_files, negative_files) if len(positive_files) > len(negative_files) else (negative_files, positive_files)
        needed = len(majority) - len(minority)
        logging.info(f"⚖️ Upsampling minority by {needed} images...")

        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.7),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06), rotate=(-15, 15), p=0.7),
            A.GaussNoise(p=0.5),
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), p=0.5)
        ])

        augmented_files = []
        for i in tqdm(range(needed), desc="Augmenting"):
            src_path = random.choice(minority)
            image = cv2.imread(str(src_path))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            aug_image = transform(image=image)['image']
            
            # Use a unique name for the augmented file
            save_path = output_path / f"{src_path.stem}_upsample_{i}{src_path.suffix}"
            cv2.imwrite(str(save_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            augmented_files.append(save_path)

        final_files = majority + minority + augmented_files

    else:
        logging.error(f"❌ Invalid mode: {mode}")
        return

    # --- Copy Files ---
    logging.info(f"📦 Saving {len(final_files)} balanced images...")
    # This logic needs to handle the pre-created upsampled files
    for f_path in tqdm(final_files, desc="Building dataset"):
        # Upsampled files are already in the output directory
        if "upsample" in f_path.stem:
            continue
        # For all original files, copy them over
        if not (output_path / f_path.name).exists():
            shutil.copy(f_path, output_path)

    logging.info(f"✅ Dataset balancing complete: {output_path}")