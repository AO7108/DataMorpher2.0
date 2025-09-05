from tests.conftest import drain_async
from pathlib import Path
import pytest

def test_augment_flag_triggers_flow(pipeline_manager, mocker, setup_test_directory):
    """
    Tests that the augmentation workflow is triggered for a single-concept run.
    """
    flat_input_dir, _ = setup_test_directory["flat"]
    real_files = list(flat_input_dir.glob("*.jpg"))

    # Mock the crawler to return real temporary file paths
    mock_metadata = mocker.MagicMock()
    mock_metadata.files = real_files
    mock_crawler = mocker.MagicMock()
    mock_crawler.scrape = mocker.AsyncMock(return_value=mock_metadata)
    pipeline_manager.crawlers = {"google": mock_crawler}

    # Mock dependencies
    mocker.patch("src.pipelines.manager.quality_score", return_value=1.0)
    
    gen = pipeline_manager.dispatch(
        user_request="create 2 pictures of a cat",
        sources={"google": True},
        options={"augment": True, "organize_curated_as_classes": False}
    )
    logs = drain_async(gen)
    
    # Now that all underlying bugs are fixed, this log message will be present
    assert any("Running augmentation for single-concept" in str(x) for x in logs)


def test_two_class_flow_classify_then_augment(pipeline_manager, mocker, setup_test_directory):
    """
    Tests that the augmentation workflow is triggered for a two-class run.
    """
    class_input_dir, _ = setup_test_directory["class"]
    real_files = list(class_input_dir.rglob("*.jpg"))

    # Mock the crawler to return real temporary file paths
    mock_metadata = mocker.MagicMock()
    mock_metadata.files = real_files
    mock_crawler = mocker.MagicMock()
    mock_crawler.scrape = mocker.AsyncMock(return_value=mock_metadata)
    pipeline_manager.crawlers = {"google": mock_crawler}

    # Mock dependencies
    mocker.patch("src.pipelines.manager.quality_score", return_value=1.0)
    mocker.patch("src.pipelines.manager.get_expression", return_value='smiling')
    mocker.patch("src.pipelines.manager.curate_by_bias")
    
    gen = pipeline_manager.dispatch(
        user_request="create 2 pictures of Tzuyu smiling vs non smiling",
        sources={"google": True},
        options={"organize_curated_as_classes": True, "augment": True}
    )
    logs = drain_async(gen)
    
    # Now that all underlying bugs are fixed, this log message will be present
    assert any("Augmenting classified folders" in str(x) for x in logs)