import asyncio
import json
import zipfile
import io
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import streamlit as st
from PIL import Image

from src.pipelines.manager import PipelineManager
from src.utils.models import DatasetMetadata
from trainers.simple_classifier import train_classifier

# --- All helper functions (init_logger, run_pipeline_stream, etc.) remain the same ---
def init_logger():
    logger = logging.getLogger("streamlit_app")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

async def run_pipeline_stream(
    request: str,
    manager: PipelineManager,
    sources: Dict[str, bool],
    options: Dict[str, Any],
    log_placeholder,
):
    final_metadata = None
    logs: List[str] = []

    async for output in manager.dispatch(request, sources, options):
        if isinstance(output, str):
            logs.append(output)
            log_placeholder.text("\n".join(logs[-200:]))
        elif isinstance(output, DatasetMetadata):
            final_metadata = output

    return final_metadata, logs

def collect_sample_images(final_stage_dir: Path, limit: int = 12) -> List[Path]:
    if not final_stage_dir or not final_stage_dir.exists():
        return []
    files = []
    for p in final_stage_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            files.append(p)
            if len(files) >= limit:
                break
    return files

def read_quality_report(report_path: Path) -> Optional[Dict[str, Any]]:
    if not report_path.exists():
        return None
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def find_run_dirs(outputs_root: Path, subject: str) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    run_dirs = sorted([d for d in outputs_root.iterdir() if d.is_dir() and d.name.startswith(subject.replace(" ", "_"))], reverse=True)
    if not run_dirs:
        return None, None, None
    run_dir = run_dirs[0]

    last_dir = None
    for name in ["split", "classes", "balanced", "curated", "augmented", "work"]:
        candidate = run_dir / name
        if candidate.exists():
            last_dir = candidate
            break
    else:
        last_dir = None

    zip_path = run_dir / "dataset.zip"
    return run_dir, last_dir, (zip_path if zip_path.exists() else None)

def has_class_folders(split_dir: Path) -> bool:
    if not split_dir.exists(): return False
    train_dir = split_dir / "train"
    if not train_dir.exists(): return False
    classes = [d for d in train_dir.iterdir() if d.is_dir()]
    return len(classes) >= 2
# --- End of helper functions ---

st.set_page_config(page_title="DataMorpher 2.0", layout="wide")
st.title("DataMorpher 2.0 — Dataset Builder")

# --- Initialize Session State ---
if "last_run_subject" not in st.session_state:
    st.session_state.last_run_subject = None
if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "last_run_options" not in st.session_state:
    st.session_state.last_run_options = {}
if "training_results" not in st.session_state:
    st.session_state.training_results = None


# Initialize a manager instance here to get the list of all possible crawlers for the UI
temp_manager = PipelineManager()
all_crawler_names = ["None"] + list(temp_manager.all_crawlers.keys())

with st.sidebar:
    st.header("Pipeline Controls")

    prompt_default = "create 40 pictures of a siamese cat"
    user_prompt = st.text_area("Enter your prompt", value=prompt_default, height=90)

    st.subheader("Data Source")
    default_crawler_index = all_crawler_names.index("google") if "google" in all_crawler_names else 0
    selected_crawler = st.selectbox(
        "Select active crawler",
        options=all_crawler_names,
        index=default_crawler_index,
        help="Choose the data source to fetch from. Select 'None' to skip crawling."
    )

    if selected_crawler in ["instagram", "tiktok", "twitter"]:
        crawler_instance = temp_manager.all_crawlers.get(selected_crawler)
        if crawler_instance and hasattr(crawler_instance, 'enabled') and not crawler_instance.enabled:
            st.warning(f"The '{selected_crawler}' crawler is currently disabled in the backend.")

    st.subheader("1. Select Your Goal")
    workflow_mode = st.radio(
        "What kind of dataset do you want to build?",
        ("Filter a Single Concept", "Build a Two-Class Dataset (e.g., A vs B)"),
        index=0,
        help="""
        - **Filter a Single Concept**: Use this to clean a dataset by keeping only images that match one idea (e.g., 'smiling'). You can optionally balance the results.
        - **Build a Two-Class Dataset**: Use this for prompts like 'smiling vs non smiling'. This will automatically create two folders, one for each class.
        """
    )
    is_class_builder_mode = (workflow_mode == "Build a Two-Class Dataset (e.g., A vs B)")

    st.subheader("2. Configure Processors")
    
    do_augment = st.checkbox("Augment", value=True, help="Create modified versions of images to increase dataset size.")
    do_split = st.checkbox("Split", value=True, help="Split the final dataset into train/val/test folders.")
    do_score = st.checkbox("Score quality", value=True, help="Analyze and report on the quality of the final images.")
    augment_after_split = st.checkbox("Augment after split (opt-in)", value=False, help="Adds augmented images in each split subset. Beware potential leakage for strict evaluation.")

    st.markdown("---")

    do_curate = st.checkbox(
        "Filter with CLIP (Curate)",
        value=False,
        disabled=is_class_builder_mode,
        help="DISABLED in Two-Class mode. Use this only to filter for a single concept (e.g., keep only smiling photos)."
    )

    do_balance = st.checkbox(
        "Balance a Filtered Dataset",
        value=False,
        disabled=is_class_builder_mode,
        help="DISABLED in Two-Class mode. Use this after filtering for a single concept to equalize counts."
    )

    st.subheader("Quality")
    min_quality = st.selectbox("Min quality band", ["low", "medium", "high"], index=1)
    
    with st.expander("Settings for 'Filter a Single Concept'"):
        st.caption("These settings are only used for the 'Filter a Single Concept' workflow.")
        curate_threshold = st.slider("Curation threshold", 0.0, 1.0, 0.25, 0.05, disabled=is_class_builder_mode)
        curate_neutral = st.text_input("Neutral prompt", value="a person with a neutral expression", disabled=is_class_builder_mode)
        balance_mode = st.radio("Balancing Strategy", options=["downsample", "upsample", "recrawl"], index=0, disabled=is_class_builder_mode)

    with st.expander("Settings for 'Build a Two-Class Dataset'"):
        st.caption("These settings are only used for the 'Build a Two-Class Dataset' workflow.")
        
        # // --- NEW: Add the Curation Engine selector ---
        curation_engine = st.selectbox(
            "Curation Engine",
            ("CLIP (General Purpose)", "DeepFace (Faces Only)"),
            index=0,
            disabled=not is_class_builder_mode,
            help="Choose 'DeepFace' for highly accurate face analysis (e.g., smiling vs neutral). Use 'CLIP' for all other tasks (e.g., cat colors)."
        )
        

        col_a, col_b = st.columns(2)
        with col_a:
            label_a = st.text_input("Label A", value="smiling", disabled=not is_class_builder_mode)
        with col_b:
            label_b = st.text_input("Label B", value="non smiling", disabled=not is_class_builder_mode)


    with st.expander("Recrawl options (for minority class)"):
        st.caption("Used when Balance mode is 'recrawl' OR for the Two-Class builder.")
        minority_prompts_str = st.text_input("Minority prompts (semicolon-separated)", value="not smiling; neutral face; serious expression")
        recrawl_gap_factor = st.slider("Recrawl intensity (fraction of gap)", 0.5, 2.0, 1.0, 0.1)
        recrawl_cap = st.number_input("Recrawl max fetch cap", 10, 500, 100, 10)

    st.subheader("Crawler settings")
    max_workers = st.slider("Max parallel downloads", 2, 24, 8, 1)
    timeout_s = st.slider("Per-download timeout (s)", 5, 60, 20, 5)
    
    # // --- CHANGED: BASELINE TRAINING SECTION LOGIC ---
    # The disabling condition is now more comprehensive.
    st.subheader("Baseline training (optional)")
    
    # Define the condition for disabling the training section
    training_disabled = (not do_split) or do_curate or do_balance
    reason = ""
    if not do_split:
        reason = "You must enable the 'Split' processor to train a model."
    elif do_curate or do_balance:
        reason = "Training is disabled when using the 'Filter' or 'Balance' options. Training is only available for the 'Two-Class Dataset Builder' workflow or simple split-only datasets."

    if training_disabled:
        st.info(reason)

    train_enable = st.checkbox("Train a baseline classifier after split", value=False, disabled=training_disabled)
    train_backbone = st.selectbox("Backbone", ["resnet18", "mobilenet_v3_small"], index=0, disabled=training_disabled)
    train_epochs = st.number_input("Epochs", 1, 50, 5, 1, disabled=training_disabled)
    train_batch = st.number_input("Batch size", 4, 128, 32, 2, disabled=training_disabled)
    train_lr = st.number_input("Learning rate", 1e-5, 1.0, 1e-3, 1e-4, format="%.5f", disabled=training_disabled)
    # // --- END CHANGED ---
    
    run_button = st.button("Run full pipeline", type="primary")

# --- The rest of the file (the `if run_button:` block and display logic) remains the same ---
# Output placeholders
log_box = st.empty()
status_box = st.empty()
preview_box = st.empty()
report_box = st.empty()
download_box = st.empty()
train_box = st.empty()

if run_button:
    if not user_prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    # Clear previous run state from session
    st.session_state.last_run_subject = None
    st.session_state.last_run_dir = None
    st.session_state.last_run_options = {}
    st.session_state.training_results = None

    # Clear UI containers
    log_box.empty()
    status_box.empty()
    preview_box.empty()
    report_box.empty()
    download_box.empty()
    train_box.empty()

    logger = init_logger()
    status_box.info("Initializing pipeline...")

    manager = temp_manager
    manager.min_quality_band = min_quality
    manager.crawler_overrides = {"max_workers": max_workers, "timeout_s": timeout_s}

    sources = {crawler: (crawler == selected_crawler) for crawler in temp_manager.all_crawlers.keys()}
    if selected_crawler == "None":
        logger.info("No crawler selected. Skipping data collection.")
    else:
        logger.info(f"Active crawler selected: {selected_crawler}")

    minority_prompts = [s.strip() for s in minority_prompts_str.split(";") if s.strip()]
    
    is_class_builder = (workflow_mode == "Build a Two-Class Dataset (e.g., A vs B)")
    class_labels = [label_a.strip(), label_b.strip()] if is_class_builder else None
    
    options = {
        "augment": do_augment,
        "split": do_split,
        "score": do_score,
        "augment_after_split": augment_after_split,
        "organize_curated_as_classes": is_class_builder,
        "class_labels": class_labels,
        "curation_engine": curation_engine,

        "curate": do_curate and not is_class_builder,
        "balance": do_balance and not is_class_builder,
        "organize_curated_as_classes": is_class_builder,
        "class_labels": class_labels,
        
        "curate_prompt": "smiling",
        "curate_neutral": curate_neutral,
        "curate_threshold": float(curate_threshold),
        "balance_positive": "smiling",
        "balance_negative": curate_neutral,
        "balance_mode": balance_mode if not is_class_builder else "recrawl",
        
        "recrawl_minority_prompts": minority_prompts,
        "recrawl_gap_factor": float(recrawl_gap_factor),
        "recrawl_cap": int(recrawl_cap),
        
        "train": train_enable,
        "train_cfg": {
            "epochs": train_epochs,
            "batch_size": train_batch,
            "lr": train_lr,
            "backbone": train_backbone,
            "patience": 3,
        },
    }

    status_box.info(f"Running pipeline with '{selected_crawler}' crawler...")
    final_metadata, logs = asyncio.run(run_pipeline_stream(user_prompt, manager, sources, options, log_box))

    if final_metadata:
        status_box.success("Pipeline finished successfully.")
        
        outputs_root = Path("outputs")
        run_dir, _, _ = find_run_dirs(outputs_root, final_metadata.subject)
        
        st.session_state.last_run_subject = final_metadata.subject
        st.session_state.last_run_dir = str(run_dir) if run_dir else None
        st.session_state.last_run_options = options
        
        if options.get("train"):
            with train_box.container():
                if not options.get("split"):
                    st.warning("Training requires a split dataset. Enable 'Split' and re-run.")
                    st.session_state.training_results = {"skipped": True, "reason": "Split disabled"}
                else:
                    split_dir = run_dir / "split" if run_dir else None
                    if not split_dir or not split_dir.exists():
                        st.warning("Split directory not found. Re-run with 'Split' enabled.")
                        st.session_state.training_results = {"skipped": True, "reason": "Split directory not found"}
                    else:
                        models_dir = run_dir / "models"
                        st.info("Training baseline classifier...")
                        try:
                            result = train_classifier(
                                data_dir=str(split_dir),
                                out_dir=str(models_dir),
                                config=options.get("train_cfg", {}),
                            )
                            st.session_state.training_results = result
                        except Exception as e:
                            st.error(f"Training failed: {e}")
                            st.session_state.training_results = {"skipped": True, "reason": f"Execution error: {e}"}
    else:
        status_box.warning("Pipeline finished without producing a dataset. Check logs above.")

if st.session_state.get("last_run_dir"):
    subject = st.session_state.last_run_subject
    run_dir = Path(st.session_state.last_run_dir)
    options = st.session_state.last_run_options
    
    _, last_stage_dir, zip_path = find_run_dirs(Path("outputs"), subject)

    with preview_box.container():
        if last_stage_dir and last_stage_dir.exists():
            sample_paths = collect_sample_images(last_stage_dir, limit=12)
            if sample_paths:
                st.subheader("Sample previews from final stage")
                cols = st.columns(4)
                for i, p in enumerate(sample_paths):
                    try:
                        img = Image.open(p).convert("RGB")
                        cols[i % 4].image(img, caption=p.name, use_column_width=True)
                    except Exception:
                        continue

    with report_box.container():
        if run_dir:
            report_path = run_dir / "quality_report.json"
            if report_path.exists():
                report = read_quality_report(report_path)
                if report:
                    st.subheader("Quality report (summary)")
                    summary = report.get("summary", {})
                    st.json(summary)
                    
                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Download full quality_report.json",
                            data=f.read(),
                            file_name=report_path.name,
                            mime="application/json",
                            key="quality_report_dl"
                        )

    with download_box.container():
        if zip_path and zip_path.exists():
            st.subheader("Download Final Dataset")
            with open(zip_path, "rb") as f:
                st.download_button(
                    label=f"Download dataset.zip ({zip_path.stat().st_size // 1024} KB)",
                    data=f.read(),
                    file_name=zip_path.name,
                    mime="application/zip",
                )

        if run_dir and run_dir.exists():
            st.markdown("---")
            st.subheader("Download Individual Pipeline Stages")

            def zip_directory_to_bytes(dir_path: Path) -> bytes:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file in dir_path.rglob('*'):
                        if file.is_file():
                            zf.write(file, arcname=file.relative_to(dir_path))
                buffer.seek(0)
                return buffer.getvalue()

            stage_dirs = ["work", "augmented", "curated", "balanced", "classes", "split"]
            for stage_name in stage_dirs:
                stage_path = run_dir / stage_name
                if stage_path.exists() and stage_path.is_dir() and any(stage_path.iterdir()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        num_files = sum(1 for f in stage_path.rglob('*') if f.is_file())
                        st.write(f"**Stage:** `{stage_name}` ({num_files} files)")
                    with col2:
                        zip_bytes = zip_directory_to_bytes(stage_path)
                        st.download_button(
                            label=f"Download .zip",
                            data=zip_bytes,
                            file_name=f"{run_dir.name}_{stage_name}.zip",
                            mime="application/zip",
                            key=f"download_{stage_name}"
                        )

    with train_box.container():
        if options.get("train", False):
            results = st.session_state.get("training_results")
            if results:
                if not results.get("skipped"):
                    model_path = Path(results["model_path"])
                    metrics_path = Path(results["metrics_path"])
                    best_acc = float(results.get("best_val_acc", 0.0))
                    st.success(f"Training complete. Best val acc: {best_acc:.3f}")

                    if model_path.exists():
                        with open(model_path, "rb") as mf:
                            st.download_button(label=f"Download model ({model_path.name})", data=mf.read(), file_name=model_path.name, mime="application/octet-stream", key="model_dl")
                    if metrics_path.exists():
                        with open(metrics_path, "rb") as jf:
                            st.download_button(label=f"Download metrics ({metrics_path.name})", data=jf.read(), file_name=metrics_path.name, mime="application/json", key="metrics_dl")
                else:
                    st.info(f"Training skipped: {results.get('reason')}")
                   