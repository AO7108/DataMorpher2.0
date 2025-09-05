# Location: src/processors/dataset_splitter.py

from pathlib import Path
import shutil
import random
import logging

# --- Logging Setup ---
logging.basicConfig(format="%(asctime)s - [%(levelname)s] - %(message)s", level=logging.INFO)

def split_dataset(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Splits a dataset into train/val/test sets. Supports both flat and class-subfolder datasets.
    Ensures class folders exist in each split subset even if empty (for class-based datasets).
    """
    # Validate ratios
    total_ratio = round(train_ratio + val_ratio + test_ratio, 4)
    if total_ratio != 1.0:
        logging.error(f"❌ Ratios must sum to 1.0 (Got {total_ratio})")
        return

    random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logging.error(f"❌ Input directory not found: {input_path}")
        return

    # Reset output folder
    if output_path.exists():
        shutil.rmtree(output_path)
    (output_path / "train").mkdir(parents=True)
    (output_path / "val").mkdir(parents=True)
    (output_path / "test").mkdir(parents=True)

    # Detect if dataset is class-based (has subfolders)
    subfolders = [f for f in input_path.iterdir() if f.is_dir()]
    is_class_based = len(subfolders) > 0

    if is_class_based:
        logging.info(f"📂 Class-based dataset detected: {len(subfolders)} classes found.")
        for class_dir in subfolders:
            _split_single_class(class_dir, output_path, train_ratio, val_ratio, test_ratio)
    else:
        logging.info("📂 Flat dataset detected.")
        _split_single_class(input_path, output_path, train_ratio, val_ratio, test_ratio, flat=True)

    logging.info(f"✅ Dataset split complete. Output at: {output_path}")

def _split_single_class(class_path, output_path, train_ratio, val_ratio, test_ratio, flat=False):
    """Helper: Splits one class folder or a flat dataset."""
    image_files = [f for f in class_path.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
    if not image_files:
        logging.warning(f"⚠️ No images found in {class_path}")
        # Still ensure empty class folders are created for class-based
        class_name = class_path.name if not flat else ""
        for split_name in ("train", "val", "test"):
            split_dir = output_path / split_name / class_name if class_name else output_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
        return

    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    class_name = class_path.name if not flat else ""
    for split_name, files in splits.items():
        split_dir = output_path / split_name / class_name if class_name else output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        # Always create class folder, even if files is empty (for class-based)
        for f in files:
            shutil.copy(f, split_dir / f.name)

    logging.info(
        f"📊 {class_name or 'Dataset'}: {total} total → "
        f"{len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test"
    )
