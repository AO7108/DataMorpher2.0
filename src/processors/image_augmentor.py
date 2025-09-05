# Location: src/processors/image_augmentor.py

import os
import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm

def augment_images(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5
):
    """
    Takes a directory of images and creates augmented versions of them.

    Args:
        input_dir: The folder containing the original images.
        output_dir: The folder where augmented images will be saved.
        augmentations_per_image: The number of new versions to create for each original image.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output directory created at: {output_path}")

    # ✅ Updated transforms to remove warnings
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent={"x": 0.06, "y": 0.06}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.GaussNoise(p=0.5),
        A.CoarseDropout(
            p=0.5
        )
    ])

    image_files = list(input_path.glob("*.[jp][pn]g"))  # Finds .jpg, .jpeg, .png
    print(f"🚀 Found {len(image_files)} images to augment. Starting process...")

    for image_path in tqdm(image_files, desc="Augmenting Images"):
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(augmentations_per_image):
            augmented = transform(image=image)
            augmented_image = augmented["image"]

            new_filename = f"{image_path.stem}_aug_{i}{image_path.suffix}"
            save_path = output_path / new_filename

            cv2.imwrite(str(save_path), cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

    print(f"✅ Augmentation complete. {len(image_files) * augmentations_per_image} new images saved.")