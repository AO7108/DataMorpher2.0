# Location: trainers/simple_classifier.py

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import shutil
from PIL import Image, ImageOps

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def _make_dataloaders(split_dir: Path, batch_size: int) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    # Standard ImageNet-size transforms
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    return train_loader, val_loader, idx_to_class


def _build_model(backbone: str, num_classes: int) -> nn.Module:
    backbone = (backbone or "resnet18").lower()
    if backbone == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m
    # default: resnet18
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def train_classifier(
    data_dir: str,
    out_dir: str,
    config: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Trains a simple classifier on outputs/<subject>/split.
    data_dir: path to split directory with train/val/test subfolders
    out_dir: directory to save model and metrics
    config:
      - epochs: int
      - batch_size: int
      - lr: float
      - backbone: 'resnet18'|'mobilenet_v3_small'
      - patience: int (early stopping)
      - is_simple_dataset: bool (if True, restructures flat dataset)
    """
    logger = logging.getLogger(__name__)
    cfg = config or {}
    epochs = int(cfg.get("epochs", 5))
    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("lr", 1e-3))
    backbone = str(cfg.get("backbone", "resnet18"))
    patience = int(cfg.get("patience", 3))
    is_simple_dataset = cfg.get("is_simple_dataset", False)

    split_dir = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    test_dir = split_dir / "test"

    # --- MODIFICATION START: Restructure the entire function ---
    # Define variables needed for cleanup before the try block
    dummy_class_name = "_negative"
    subject_name = ""

    try:
        if is_simple_dataset:
            logger.info("Simple dataset detected. Creating artificial negative class for training.")
            subject_name = split_dir.parent.name.split('_')[0] or "class_a"

            def restructure_and_create_dummy(flat_dir: Path, class_name: str):
                if not flat_dir.exists(): return
                
                class_dir = flat_dir / class_name
                class_dir.mkdir(exist_ok=True)
                files_to_move = [f for f in flat_dir.iterdir() if f.is_file()]
                for f in files_to_move:
                    shutil.move(str(f), str(class_dir / f.name))
                
                dummy_dir = flat_dir / dummy_class_name
                dummy_dir.mkdir(exist_ok=True)
                files_to_transform = list(class_dir.glob("*"))
                if not files_to_transform: return

                for f in files_to_transform:
                    try:
                        with Image.open(f) as img:
                            inverted_img = ImageOps.invert(img.convert("RGB"))
                            new_path = dummy_dir / f"dummy_{f.name}"
                            inverted_img.save(new_path)
                    except Exception as e:
                        logger.warning(f"Could not create dummy image from {f.name}: {e}")
            
            restructure_and_create_dummy(train_dir, subject_name)
            restructure_and_create_dummy(val_dir, subject_name)
            restructure_and_create_dummy(test_dir, subject_name)

        if not train_dir.exists() or not val_dir.exists():
            logger.warning(f"Training skipped: split folders not found at {split_dir}")
            return {"skipped": True, "reason": "no_split"}

        class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
        if len(class_dirs) < 2:
            logger.warning("Training skipped: need at least 2 classes.")
            return {"skipped": True, "reason": "single_or_no_class"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Starting training on device: {device}")

        train_loader, val_loader, idx_to_class = _make_dataloaders(split_dir, batch_size)
        num_classes = len(idx_to_class)

        model = _build_model(backbone, num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        best_val_acc = 0.0
        best_state = None
        epochs_no_improve = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type if device.type != "cpu" else "cpu", enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / max(1, total)
            train_acc = correct / max(1, total)

            val_loss, val_acc = _evaluate(model, val_loader, device, loss_fn)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            logger.info(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} acc={train_acc:.3f} | val_loss={val_loss:.4f} acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        ts = time.strftime("%Y%m%d_%H%M%S")
        model_dir = out_path
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{ts}_classifier.pt"
        metrics_path = model_dir / f"{ts}_metrics.json"

        if best_state is not None:
            model.load_state_dict(best_state)

        torch.save({
            "model_state": model.state_dict(),
            "backbone": backbone,
            "num_classes": num_classes,
            "idx_to_class": idx_to_class,
        }, model_path)

        metrics = {
            "epochs_run": len(history["val_acc"]),
            "best_val_acc": best_val_acc,
            "history": history,
            "class_index": idx_to_class,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "backbone": backbone,
                "patience": patience,
            }
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Model saved to: {model_path}")
        logger.info(f"Metrics saved to: {metrics_path}")

        return {
            "skipped": False,
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "best_val_acc": best_val_acc,
        }

    finally:
        # This block now runs AFTER the training is complete or if an error occurs.
        if is_simple_dataset:
            logger.info("Cleaning up temporary negative class directories...")
            for s_dir in [train_dir, val_dir, test_dir]:
                if not s_dir.exists(): continue
                
                dummy_dir_path = s_dir / dummy_class_name
                if dummy_dir_path.exists():
                    shutil.rmtree(dummy_dir_path)
                
                # Restore original flat structure by moving images out of the subject folder
                original_class_dir = s_dir / subject_name
                if original_class_dir.exists():
                    for f in original_class_dir.glob("*"):
                        shutil.move(str(f), str(s_dir / f.name))
                    try:
                        original_class_dir.rmdir()
                    except OSError: # Fails if not empty, which is fine
                        pass
    # --- MODIFICATION END ---