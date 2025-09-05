import pytest
from src.parser.command_parser import CommandParser

@pytest.fixture
def parser():
    """Provides a CommandParser instance for all tests."""
    return CommandParser()

def test_parse_simple_create_command(parser):
    """Tests parsing of a basic 'create dataset' command."""
    prompt = "create 20 pictures of tzuyu"
    result = parser.parse(prompt)
    assert result is not None
    assert result.get("intent") == "create_dataset"
    assert result.get("subject") == "Tzuyu"
    assert result.get("count") == 20
    assert result.get("modality") == "image"
    assert result.get("class_split") is False

def test_parse_class_split_command(parser):
    """Tests parsing of a command with class splitting and balancing hints."""
    prompt = "create 60 pictures of tzuyu smiling vs non smiling, equal"
    result = parser.parse(prompt)
    assert result is not None
    assert result.get("intent") == "create_dataset"
    assert result.get("subject") == "Tzuyu"
    assert result.get("class_split") is True
    assert result.get("class_labels") == ["smiling", "non smiling"]
    assert result.get("balance_hint") == "equal"

def test_parse_with_processors(parser):
    """Tests that processor keywords are correctly identified in the prompt."""
    prompt = "get 50 photos of cats, then curate split and score the data"
    result = parser.parse(prompt)
    assert result is not None
    assert result.get("processors") is not None
    assert "curate" in result["processors"]
    assert "split" in result["processors"]
    assert "score" in result["processors"]
    assert "balance" not in result["processors"]

def test_parse_unsupported_command(parser):
    """Tests that a non-dataset-related command is handled gracefully."""
    prompt = "what is the weather like in shimla?"
    result = parser.parse(prompt)
    assert result is not None
    assert result.get("intent") == "unsupported"

def test_parse_no_count_finds_none(parser):
    """Tests that if no number is in the prompt, the count is None."""
    prompt = "I want images of dogs"
    result = parser.parse(prompt)
    assert result is not None
    assert result.get("count") is None

def test_heuristic_subject_guess(parser):
    """Tests the fallback logic for guessing the subject when it's not explicit."""
    # The heuristic looks for a capitalized word before an attribute like 'smiling'
    prompt = "Generate some pictures of a Black Cat smiling"
    result = parser.parse(prompt)
    assert result is not None
    # Note: The heuristic capitalizes the result
    assert result.get("subject") == "Black Cat"

def test_clean_class_labels(parser):
    """Ensures class labels are cleaned of extra spaces and punctuation."""
    prompt = "a dataset of happy dogs vs. sad dogs,"
    result = parser.parse(prompt)
    # The parser is now smart enough to get the full labels.
    # We update the test to expect this new, correct result.
    assert result.get("class_labels") == ["happy dogs", "sad dogs"]