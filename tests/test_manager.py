# We need to import the function we want to test
from src.pipelines.manager import _band_from_score

# Each test is a simple function that starts with 'test_'
def test_band_from_score_returns_high_for_high_score():
    """
    Tests if a score of 0.8 or greater correctly returns 'high'.
    """
    # 1. Arrange: Set up our input data
    input_score = 0.9
    expected_output = "high"

    # 2. Act: Call the function with the input data
    actual_output = _band_from_score(input_score)

    # 3. Assert: Check if the result is what we expected
    assert actual_output == expected_output

def test_band_from_score_returns_medium_for_medium_score():
    """
    Tests if a score between 0.5 and 0.8 correctly returns 'medium'.
    """
    assert _band_from_score(0.65) == "medium"

def test_band_from_score_returns_low_for_low_score():
    """
    Tests if a score below 0.5 correctly returns 'low'.
    """
    assert _band_from_score(0.4) == "low"

def test_band_from_score_edge_cases():
    """
    Tests the exact boundary conditions.
    """
    assert _band_from_score(0.8) == "high"  # Boundary for high
    assert _band_from_score(0.5) == "medium" # Boundary for medium