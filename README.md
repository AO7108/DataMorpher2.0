# DataMorpher 2.0
<!-- Hero Image: replace src with your banner; remove width or adjust as needed -->
<p align="center">
  <img src="c:\AI\DataMorpher2.0\my-banner.png" alt="DataMorpher 2.0" width="820" />
</p>

<!-- Badges: update as desired -->
<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+" /></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" /></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" /></a>
  <a href="#"><img src="https://img.shields.io/badge/OS-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="OS" /></a>
  <a href="#development-and-tests"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome" /></a>
</p>

A modular, production-oriented dataset builder for images, audio, and text. DataMorpher orchestrates multi-source crawling, intelligent curation (CLIP or DeepFace), augmentation, balancing, splitting, and quality scoring, and can optionally train a simple baseline classifier. Operate it via a CLI driver or an interactive Streamlit UI.

- Crawl sources: Google Images (SerpApi), Reddit (PRAW), YouTube (yt-dlp; optional audio via ffmpeg)
- Curate/filter: CLIP-based relevance (general-purpose) or DeepFace emotion analysis (faces only, smiling vs non-smiling)
- Process: augmentation, two-class class-builder, CLIP-based balancing, split, quality scoring
- Train: quick baseline classifier (ResNet18 or MobileNetV3)
- Outputs: versioned run folders with artifacts (stages, zip, quality report, model/metrics)

This document provides a deep-dive for professional users: architecture, internals, configuration, reproducibility, performance, and extensibility.

---

## Table of Contents

- Quickstart
- Overview
- Architecture and Data Flow
- Features, Algorithms, and Internals
  - Crawlers
  - Curation Engines
  - Processors
  - Training Subsystem
  - Quality Scoring
- Configuration
  - config.yaml (LLM and paths)
  - Environment Variables and Secrets
- Installation
  - Python deps, Torch notes, ffmpeg
  - Optional extras (DeepFace, Aesthetics)
- Usage
  - CLI (app.py) with full option reference
  - Streamlit UI (app_streamlit.py)
- Outputs and Run Structure
- Performance, Reliability, Reproducibility
- Troubleshooting
- Extensibility (adding crawlers/processors)
- Security, Ethics, and Compliance
- Development and Tests
- FAQ

---

## Quickstart

Prerequisites
- Python 3.10+ on Windows, Linux, or macOS
- Optional: ffmpeg for YouTube audio extraction
- Optional: API keys for SerpApi/Reddit if you want those crawlers

Setup
```
python -m venv venv
venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# CPU-only Torch (see Installation section for CUDA wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Run a minimal CLI job
```
python app.py --interactive
# Example prompt:
# create 40 pictures of a siamese cat
```

Or a quick pipeline with built-ins
```
python app.py --augment --split --score
```

Launch the UI
```
streamlit run app_streamlit.py
```

Optional: enable crawlers
Create a `.env` file in the repo root:
```
SERPAPI_KEY=...
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=YourAppName/1.0 by your_username
```

Outputs
- Datasets and artifacts will be under `outputs/<subject>_<timestamp>/`

---

## Overview

DataMorpher 2.0 transforms a natural-language request (e.g., “create 60 pictures of tzuyu smiling vs non smiling, equal; curate balance split score”) into a concrete pipeline that crawls, curates, processes, and packages datasets. It supports both a single-concept workflow (filter/curate + optional balance) and a two-class workflow (class builder: A vs B) with optional recrawl to address class imbalance.

---

## Architecture and Data Flow

Modules (under `src/`):

- Pipelines
  - `pipelines/manager.py` — Core orchestrator. Parses user intent, selects sources, runs processors, manages staging, artifacts, re-crawl, packaging, and optional training.
- Crawlers (`crawlers/`)
  - `google_images_crawler.py` (SerpApi), `reddit_crawler.py` (PRAW), `youtube_crawler.py` (yt-dlp + optional ffmpeg)
  - Stubs for Instagram/TikTok/Twitter (enabled only if dependencies and credentials exist)
- Processors (`processors/`)
  - `image_augmentor.py` (albumentations), `bias_curator.py` (CLIP/threshold), `dataset_balancer.py` (CLIP-driven balance), `dataset_splitter.py`, `quality_scorer.py` (sharpness + optional aesthetics), `face_classifier.py` (DeepFace for smiling vs non-smiling)
- Parser (`parser/command_parser.py`)
  - Natural-language to normalized command dict (heuristics + optional local LLM via llama-cpp)
- Trainer (`trainers/simple_classifier.py`)
  - Torch-based baseline classifier using torchvision (ResNet18 or MobileNetV3)
- Utilities (`utils/`)
  - Logging, models, settings, config loader
- Plugins (`plugins/`)
  - Typed plugin scaffolding for future modalities (audio/text dataset plugins)

High-level flow:
1) `PipelineManager` parses the user request via `CommandParser` (LLM-backed if configured).
2) Chooses workflow:
   - Single-concept: crawl → curate (optional) → balance (optional) → augment/split/score/train
   - Two-class: targeted crawl per class → class builder (CLIP or DeepFace) → optional recrawl to fix class imbalance → clean-up and downsample → optional augment → split/score/train
3) Stages are written to a versioned output folder: `outputs/<subject>_<timestamp>/`
4) Final packaging: zip of last stage, reports, and optional model/metrics.

---

## Features, Algorithms, and Internals

### Crawlers

All crawlers expose a common interface used by `PipelineManager`:
- `scrape(subject: str, limit: int, attributes: List[str], output_dir: Path) -> DatasetMetadata | None`
- Optional: `scrape_with_query(query: str, limit: int, output_dir: Path) -> List[Path]` (used for recrawl)

Available:
- Google Images (SerpApi)
  - File: `crawlers/google_images_crawler.py`
  - Dependencies: `google-search-results` (SerpApi SDK), `aiohttp`, `aiofiles`, `Pillow`
  - Behavior: builds queries using subject + attributes; respects configured timeouts/workers (via `PipelineManager.crawler_overrides`)
- Reddit (PRAW)
  - File: `crawlers/reddit_crawler.py`
  - Dependencies: `praw`, `aiohttp`, `Pillow`
  - Credentials: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`
  - Logic: searches several subreddits for suitable image posts; downloads images asynchronously; restricts to image hosts (`i.redd.it`, `i.imgur.com`) and content-type `image/*`; converts to RGB JPEGs
- YouTube (yt-dlp)
  - File: `crawlers/youtube_crawler.py`
  - Dependencies: `yt-dlp`, optional `ffmpeg-python` for audio extraction
  - Behavior: downloads up to N videos via yt-dlp (“best mp4”); optionally extracts audio to mp3 if the user requests audio (attributes include "audio"); warns if ffmpeg is not installed

Crawler enablement:
- Each crawler attempts to import its dependency; if missing or credentials absent, it marks itself disabled and `PipelineManager` excludes it.

### Curation Engines

Two alternatives:

1) CLIP (general-purpose)
   - Files: `processors/bias_curator.py`, `processors/dataset_balancer.py`
   - Model: `openai/clip-vit-base-patch32` via `transformers`
   - Usage:
     - Single-concept filtering: keep images matching prompt (with a neutral prompt for negative bias control), threshold default 0.25
     - Two-class class builder: classify images into A vs B using positive and neutral prompts with separate thresholds
     - Dataset balancing: uses CLIP logits to classify images against `(positive_prompt, negative_prompt)`, then downsample or augment the minority
   - Device selection: auto (“cuda” if available else “cpu”)

2) DeepFace (faces only)
   - File: `processors/face_classifier.py`
   - Dependencies: `deepface` (optional)
   - Logic: `DeepFace.analyze(..., actions=['emotion'], detector_backend='retinaface')`
   - Mapping: dominant_emotion == "happy" → “smiling”; else → “non smiling”
   - Used in two-class builder when “DeepFace (Faces Only)” engine is selected

Curation thresholds and neutral prompt (CLIP):
- Default positive threshold: 0.25
- For negative folder exclusion checks and bias control, the code uses a neutral prompt (e.g., “a person with a neutral expression, neutral gaze, relaxed lips, no grin, no visible teeth”)

### Processors

- Augmentation
  - File: `processors/image_augmentor.py`
  - Library: `albumentations`
  - Used as a standalone stage, and for upsampling in the balancer
- Dataset Balancer (CLIP)
  - File: `processors/dataset_balancer.py`
  - Classifies with CLIP and either:
    - Downsamples majority to match minority, or
    - Upsamples minority with augmentation (horizontal flips, brightness/contrast, affine, noise, random crop)
- Dataset Splitter
  - File: `processors/dataset_splitter.py`
  - Produces `split/train`, `split/val`, `split/test` (standard folder structure)
- Quality Scorer
  - File: `processors/quality_scorer.py`
  - Measures:
    - Sharpness: Laplacian variance mapped to [0, 1] using min/max bounds (50.0–1000.0)
    - Optional aesthetics: if `pyiqa` available, computes NIMA score (~1–10), normalized to [0, 1]; unified quality = 0.7*sharpness + 0.3*aesthetics
  - Outputs `quality_report.json` with summary and per-image metrics
- Two-Class Builder
  - File: `pipelines/manager.py` (logic), uses `bias_curator.py` or `face_classifier.py`
  - Pipeline:
    1) Crawl more than needed (overshoot)
    2) Curate into two folders using CLIP or DeepFace
    3) Optional recrawl of minority class (see below)
    4) Cleanup pass (remove mismatches)
    5) Downsample to equalize counts
    6) Optional augmentation per class
    7) Retag files with consistent names per class

Recrawl strategy (minority class):
- Trigger: if severe imbalance
- Process: creates temp `_recrawl_raw`, runs `scrape_with_query` with minority prompts (UI-configured), then curates only minority-consistent images (DeepFace or CLIP), copies to minority folder, deletes temp folder
- Current implementation uses `recrawl_minority_prompts` and `recrawl_cap`; UI exposes `recrawl_gap_factor` but it is not currently used by the backend logic

### Training Subsystem

- File: `trainers/simple_classifier.py`
- Torch-based quick baseline on split datasets:
  - Backbones: `resnet18`, `mobilenet_v3_small`
  - Config: epochs, batch_size, lr, patience, etc.
  - Outputs: `models/` folder under the run directory, model file(s) and a `metrics.json` including best validation accuracy
- Training enablement rules (UI):
  - Requires Split to be enabled
  - Disabled when Curate/Balance are used in single-concept mode (training is intended for two-class builder or simple split-only datasets)

---

## Configuration

### config.yaml

```
paths:
  outputs: "outputs"
  data_raw: "data/raw"
  cache: ".cache"

llm:
  model_path: "C:/AI/DataMorpher2.0/models/phi-3-mini-4k-instruct.Q4_K_M.gguf"
  n_ctx: 4096
  temperature: 0.1
  max_tokens: 1024
  n_threads: 4
```

- LLM (optional, for `CommandParser`)
  - Uses `llama-cpp-python` to parse free-text into structured commands
  - If unavailable or fails to load, the parser falls back to robust heuristics
  - The parser truncates generation using a “}” stop token and appends a brace if missing to ensure valid JSON

### Environment Variables and Secrets

Create `.env` in the project root. This file is ignored by git.

Required for specific crawlers:
- SerpApi (Google Images)
  - `SERPAPI_KEY=...`
- Reddit (PRAW)
  - `REDDIT_CLIENT_ID=...`
  - `REDDIT_CLIENT_SECRET=...`
  - `REDDIT_USER_AGENT=YourAppName/1.0 by your_username`

Optional:
- Twitter, Instagram, TikTok tokens (future integrations)
- `GEMINI_API_KEY` or other LLM APIs if you extend the project

Never commit `.env`. This repository’s `.gitignore` is set to ignore secrets and large artifacts.

---

## Installation

Python 3.10+ recommended. Windows, Linux, or macOS.

1) Virtual environment
```
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

2) Install dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

3) PyTorch/TorchVision (choose the correct wheel)
- CPU-only:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
- CUDA 12.1 (example):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Refer to https://pytorch.org/get-started/locally/

4) ffmpeg (for YouTube audio extraction)
- Windows (scoop): `scoop install ffmpeg`
- macOS (brew): `brew install ffmpeg`
- Linux (apt): `sudo apt-get install ffmpeg`

5) Optional extras
- DeepFace (face-based curation in class builder): installed via `requirements.txt` (may pull additional backends; installation may take time)
- Aesthetic scoring:
  - If you want the aesthetics component in `quality_scorer.py`, install:
    - `pyiqa` (adds NIMA and other metrics)
    - `timm` (some `pyiqa` metrics depend on timm)
  - Not required for baseline sharpness scoring; code gracefully degrades if missing

---

## Usage (CLI)

Run with defaults and built-in sample prompts:
```
python app.py
```

Interactive single-run mode:
```
python app.py --interactive
# Example input:
# create 40 pictures of a siamese cat
```

Full option reference (from `app.py`):
- General
  - `--min-quality {low,medium,high}` (default: medium). Scoring bands map from sharpness-based quality: high (≥0.8), medium (≥0.5), else low.
  - `--interactive` prompt for a custom single request
- Processors
  - `--augment` run augmentation
  - `--curate` CLIP-based filtering (single-concept only)
  - `--balance` CLIP-based balancing (single-concept only)
  - `--split` produce `train/val/test`
  - `--score` produce `quality_report.json`
- Two-class builder
  - `--classes-enable` force two-class organization
  - `--class-labels "LabelA|LabelB"` set class labels
- Training
  - `--train`
  - `--train-epochs 5` (default 5)
  - `--train-batch-size 32` (default 32)
  - `--train-lr 1e-3` (default 1e-3)
  - `--train-backbone {resnet18, mobilenet_v3_small}` (default resnet18)
- Crawler reliability/controls
  - `--max-workers 8` (default 8)
  - `--timeout 20` seconds (default 20)
  - `--curate-threshold 0.25` (default)
  - `--balance-mode {downsample, upsample}` (default downsample)

Examples:
```
# Single-concept: augment + split + score
python app.py --augment --split --score

# Two-class builder (smiling vs non smiling), split + score
python app.py --classes-enable --class-labels "smiling|non smiling" --split --score

# Baseline training (requires split)
python app.py --split --train --train-epochs 10 --train-backbone resnet18
```

---

## Usage (Streamlit UI)

Launch:
```
streamlit run app_streamlit.py
```

Key panels:
- Data Source
  - Choose one crawler (Google/Reddit/YouTube). “None” skips crawling.
- Workflow Mode
  - Filter a Single Concept
  - Build a Two-Class Dataset (A vs B)
- Processors
  - Augment, Split, Score; “Augment after split” is opt-in (use with care to avoid leakage for strict evaluation)
- Quality
  - Min band (low/medium/high). Affects acceptance after initial crawl by mapping sharpness to bands.
- Single-concept settings
  - `curate_threshold` (default 0.25)
  - `curate_neutral` prompt for bias control
  - `balance_mode` (downsample/upsample/recrawl)
- Two-class settings
  - Curation Engine: `CLIP (General Purpose)`, `DeepFace (Faces Only)`
  - Label A / Label B (defaults: “smiling”, “non smiling”)
- Recrawl options
  - `minority_prompts` (semicolon-separated)
  - `recrawl_gap_factor` (UI-exposed; not currently used by backend)
  - `recrawl_cap` (max items to fetch across recrawl queries)
- Crawler settings
  - `max_workers`, `timeout_s`
- Baseline training (optional)
  - Enabled only when Split is on and Curate/Balance (single-concept) are off

UI outputs:
- Live logs
- Sample previews from the final stage
- Download buttons for the entire dataset (`dataset.zip`) and for individual stages
- If training ran: downloadable model and metrics

---

## Outputs and Run Structure

Each run creates `outputs/<subject_slug>_<YYYYMMDD_HHMMSS>/`. Common contents:

- `work/` — kept files after initial crawl and quality filter
- `augmented/` — single-concept augmentation results
- `curated/`, `balanced/` — single-concept curation/balancing
- `classes/` — two-class builder folders
  - Final cleanup and downsampling ensure A/B parity
- `split/` — `train/`, `val/`, `test/` (class folders if two-class)
- `models/` — classifier checkpoints and metrics (if training ran)
- `quality_report.json` — optional quality report (sharpness + aesthetics if enabled)
- `dataset.zip` — packaged last-stage contents (post-split or last produced stage)

File naming:
- `_retag_files` enforces consistent naming: `<label>_<index>.ext` after classification

---

## Performance, Reliability, Reproducibility

- Device and models
  - CLIP runs on CUDA if available, else CPU
  - Training performance depends on GPU availability
- Crawler concurrency/timeouts
  - Configure via CLI (`--max-workers`, `--timeout`) or in UI
- Threshold tuning (CLIP)
  - Start at 0.25; increase for stricter inclusion, reduce to be more permissive
- Recrawl behavior
  - Recrawl aims to fill minority shortfall using targeted prompts; backend uses `recrawl_cap` and distributes per-query
- Quality bands (‘low’, ‘medium’, ‘high’)
  - Map from normalized Laplacian variance (sharpness). Internal thresholds: high ≥ 0.8, medium ≥ 0.5, else low
- Reproducibility
  - Crawling and augmentation introduce nondeterminism (network, randomness)
  - To improve determinism:
    - Use stable prompts and fix seeds in augmentation/training if you extend code
    - Cache data locally where possible
- Windows path length
  - Prefer short base paths (e.g., `C:\AI\DataMorpher2.0`)

---

## Troubleshooting

- PyTorch install fails or incompatible
  - Install wheels from the official index for your OS/CUDA as shown above
- ffmpeg not found
  - Install system ffmpeg; ensure it’s on PATH for `ffmpeg-python` to work
- SerpApi errors or zero results
  - Check `SERPAPI_KEY`, quotas, and query attributes
- Reddit disabled
  - Ensure all three env vars (`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`)
- DeepFace heavy install
  - Optional; only needed for face-based class builder. Use CLIP engine otherwise
- Aesthetic scoring not active
  - Install `pyiqa` (and `timm` if required); otherwise only sharpness is used
- Training disabled in UI
  - Split must be enabled; training is disabled when Curate/Balance are used in single-concept mode

---

## Extensibility

Add a new crawler:
- Create `src/crawlers/<your_crawler>.py` exposing:
  - `scrape(subject, limit, attributes, output_dir) -> DatasetMetadata|None`
  - Optional `scrape_with_query(query, limit, output_dir) -> List[Path]`
- Add dependency checks inside the crawler; set `self.enabled=False` if missing
- Register in `PipelineManager.__init__` under `self.all_crawlers`

Add a new processor:
- Implement a pure function in `src/processors/` that takes input/output dirs and parameters
- Integrate in `PipelineManager` around the appropriate stage, following the existing staging convention

Swap curation models:
- For CLIP-based methods, change `model_id` in `dataset_balancer.py` or update `bias_curator.py` to use a different transformer

---

## Security, Ethics, and Compliance

- Do not commit API keys or `.env` files
- Respect terms of service of crawled platforms and applicable laws
- Review and curate datasets for bias, privacy, and ethical considerations
- The two-class builder provides options (DeepFace vs CLIP) to reduce misclassification for face-related tasks, but human review is still advised

---

## Development and Tests

- Style: standard Python conventions
- Tests:
```
pytest -q
```
- Utilities:
  - `debug_parser.py` — introspect parser behavior (e.g., “happy dogs vs. sad dogs”)

---

## FAQ

- Q: Do I need DeepFace?
  - No. Use CLIP engine for general-purpose classification. DeepFace is only recommended for facial expression tasks.
- Q: Torch install fails on Windows?
  - Use the official wheel with `--index-url` matching your CUDA or CPU-only setup.
- Q: Where are my datasets?
  - Under `outputs/<subject>_<timestamp>/`. Use the zip or stage downloads in the UI.
- Q: Why is training disabled in UI?
  - Enable Split; ensure Curate/Balance are off in single-concept mode. Training is best used for two-class datasets or simple split-only runs.
- Q: How do I turn on aesthetic scoring?
  - Install `pyiqa` (and possibly `timm`). The code automatically enables aesthetics if available.
