import pytest
from pathlib import Path
import shutil
# Note: cv2 and numpy imports are no longer needed here as the fixture is in conftest.py

# Import the functions to be tested
from src.processors.dataset_splitter import split_dataset
from src.processors.image_augmentor import augment_images

#
# The old @pytest.fixture for setup_test_directory has been DELETED from this file.
# Pytest will now automatically find and use the correct one from tests/conftest.py.
#

def test_dataset_splitter_flat(setup_test_directory):
    """Tests splitting of a flat directory of 100 images."""
    input_dir, output_dir = setup_test_directory["flat"]
    
    split_dataset(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42
    )
    
    train_files = list((output_dir / "train").glob("*.jpg"))
    val_files = list((output_dir / "val").glob("*.jpg"))
    test_files = list((output_dir / "test").glob("*.jpg"))
    
    assert len(train_files) == 70
    assert len(val_files) == 20
    assert len(test_files) == 10

def test_dataset_splitter_class_based(setup_test_directory):
    """Tests splitting of a directory with class subfolders."""
    input_dir, output_dir = setup_test_directory["class"]
    
    # The splitter will look for subdirectories inside the input_dir
    split_dataset(str(input_dir), str(output_dir), train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    
    # Check class A counts (50 images total)
    assert len(list((output_dir / "train" / "class_a").glob("*.jpg"))) == 40
    assert len(list((output_dir / "val" / "class_a").glob("*.jpg"))) == 5
    assert len(list((output_dir / "test" / "class_a").glob("*.jpg"))) == 5
    
    # Check class B counts (20 images total)
    assert len(list((output_dir / "train" / "class_b").glob("*.jpg"))) == 16
    assert len(list((output_dir / "val" / "class_b").glob("*.jpg"))) == 2
    assert len(list((output_dir / "test" / "class_b").glob("*.jpg"))) == 2

def test_image_augmentor(setup_test_directory):
    """Tests that the augmentor creates the correct number of new images."""
    # This test now correctly gets the "flat" dataset from the conftest fixture
    flat_input_dir, output_dir = setup_test_directory["flat"]
    
    # We'll test with a smaller subset to keep it fast
    temp_input = output_dir / "temp_input"
    temp_input.mkdir()
    
    for i in range(5):
        shutil.copy(flat_input_dir / f"image_{i}.jpg", temp_input)
    
    augment_images(
        input_dir=str(temp_input),
        output_dir=str(output_dir),
        augmentations_per_image=3
    )
    
    output_files = list(output_dir.glob("*.jpg"))
    # 5 source images * 3 augmentations each = 15 new images
    assert len(output_files) == 15