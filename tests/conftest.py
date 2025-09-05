import asyncio
import pytest
import cv2
import numpy as np

from src.pipelines.manager import PipelineManager
from pathlib import Path


@pytest.fixture
def pipeline_manager() -> PipelineManager:
    return PipelineManager()


def drain_async(gen):
    out = []
    async def _drain():
        async for item in gen:
            out.append(item)
    asyncio.get_event_loop().run_until_complete(_drain())
    return out


@pytest.fixture
def setup_test_directory(tmp_path: Path):
    """
    Creates a temporary directory structure with VALID dummy images for testing.
    """
    flat_input_dir = tmp_path / "flat_input"
    class_input_dir = tmp_path / "class_input"
    output_dir = tmp_path / "output"

    flat_input_dir.mkdir()
    class_input_dir.mkdir()
    output_dir.mkdir()

    # --- Create 100 flat files for test_dataset_splitter_flat ---
    for i in range(100):
        # Create a small, valid 10x10 black image
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(flat_input_dir / f"image_{i}.jpg"), dummy_image)

    # --- Create class-based files for test_dataset_splitter_class_based ---
    class_a_dir = class_input_dir / "class_a"
    class_b_dir = class_input_dir / "class_b"
    class_a_dir.mkdir()
    class_b_dir.mkdir()
    # Create 50 images for class_a
    for i in range(50):
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_a_dir / f"image_a_{i}.jpg"), dummy_image)
    # Create 20 images for class_b
    for i in range(20):
        dummy_image = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(str(class_b_dir / f"image_b_{i}.jpg"), dummy_image)

    # Return a dictionary of paths so each test can get what it needs
    return {
        "flat": (flat_input_dir, output_dir),
        "class": (class_input_dir, output_dir),
    }