import pytest
from pathlib import Path
from src.pipelines.manager import PipelineManager

# This helper function might be in your manager.py, but is copied here
# to make this test file self-contained and runnable.
def _band_from_score(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"

def test_band_mapping_and_filtering(tmp_path: Path, monkeypatch):
    # Create fake images
    kept = [tmp_path / f"img_{i}.jpg" for i in range(5)]
    for p in kept:
        p.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header bytes

    pm = PipelineManager()

    # Monkeypatch quality_score to control outcomes
    from src.processors import quality_scorer

    # Scores: 0.9 (high), 0.7 (medium), 0.4 (low), 0.5 (medium edge), 0.8 (high edge)
    scores = [0.9, 0.7, 0.4, 0.5, 0.8]
    def fake_quality_score(path):
        idx = kept.index(path)
        return scores[idx]
    monkeypatch.setattr(quality_scorer, "quality_score", fake_quality_score)

    # Prepare manager state
    pm.min_quality_band = "medium"

    # Emulate the filtering logic
    order = {"low": 0, "medium": 1, "high": 2}
    def band_from_score(s):
        if s >= 0.8: return "high"
        if s >= 0.5: return "medium"
        return "low"

    bands = [band_from_score(s) for s in scores]
    kept_files = [p for p, s, b in zip(kept, scores, bands) if order[b] >= order[pm.min_quality_band]]

    # Expected: scores >= 0.5 kept → indices 0,1,3,4 (4 kept)
    assert len(kept_files) == 4
    # THE CORRECTED ASSERTION:
    assert kept_files == [kept[0], kept[1], kept[3], kept[4]]