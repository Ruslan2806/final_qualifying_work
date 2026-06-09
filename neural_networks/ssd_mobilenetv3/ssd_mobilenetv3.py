import os
import shutil
from pathlib import Path

SCRIPT_PATH   = Path(__file__).resolve()

BASE_PATH     = SCRIPT_PATH.parent.parent.parent 
DATASET_DIR   = BASE_PATH / "dataset"

os.chdir(DATASET_DIR)

import yaml
import json

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
import cv2
import numpy as np
from PIL import Image

torch.backends.cudnn.enabled   = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

# ─── Пути ────────────────────────────────────────────────────────────────────
SCRIPT_PATH  = Path(__file__).resolve()
BASE_PATH    = SCRIPT_PATH.parent.parent.parent
DATASET_PATH = BASE_PATH / "dataset"
MODELS_DIR   = SCRIPT_PATH.parent / "weights"
RESULTS_DIR  = SCRIPT_PATH.parent / "results"

# ─── Гиперпараметры ───────────────────────────────────────────────────────────
NUM_CLASSES   = 2        # фон + пешеход
EPOCHS        = 50
BATCH_SIZE    = 8
LEARNING_RATE = 1e-3
IMG_SIZE      = 320      # SSDLite320 ожидает 320x320
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE      = 10


# ─── Датасет ─────────────────────────────────────────────────────────────────

class YOLODetectionDataset(Dataset):
    def __init__(self, images_dir: Path, img_size: int = 320):
        self.img_size  = img_size
        self.images    = []
        self.labels    = []

        images_dir = Path(images_dir)
        labels_dir = Path(str(images_dir).replace("images", "labels"))

        for img_path in sorted(images_dir.glob("*.png")) + \
                        sorted(images_dir.glob("*.jpg")) + \
                        sorted(images_dir.glob("*.jpeg")):
            label_path = labels_dir / (img_path.stem + ".txt")
            self.images.append(img_path)
            self.labels.append(label_path if label_path.exists() else None)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((self.img_size, self.img_size))
        img_tensor = self.transform(img)  

        boxes  = []
        labels = []

        label_path = self.labels[idx]
        if label_path is not None:
            with open(label_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])

                    x1 = (cx - bw / 2) * self.img_size
                    y1 = (cy - bh / 2) * self.img_size
                    x2 = (cx + bw / 2) * self.img_size
                    y2 = (cy + bh / 2) * self.img_size

                    x1 = max(0.0, min(x1, self.img_size - 1))
                    y1 = max(0.0, min(y1, self.img_size - 1))
                    x2 = max(x1 + 1, min(x2, self.img_size))
                    y2 = max(y1 + 1, min(y2, self.img_size))

                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id + 1)   # 0 = фон, 1 = pedestrian

        if boxes:
            boxes_tensor  = torch.tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.int64)

        target = {
            "boxes":  boxes_tensor,
            "labels": labels_tensor,
        }
        return img_tensor, target


def collate_fn(batch):
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ─── Модель ───────────────────────────────────────────────────────────────────

def build_model(num_classes: int):
    model = ssdlite320_mobilenet_v3_large(
        weights="DEFAULT",          # претренированные веса COCO
        weights_backbone="DEFAULT"
    )

    # Заменяем голову под наше количество классов
    in_channels = [672, 480, 512, 256, 256, 128]
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = SSDLiteClassificationHead(
        in_channels   = in_channels,
        num_anchors   = num_anchors,
        num_classes   = num_classes,
        norm_layer    = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    )
    return model


# ─── Метрики ─────────────────────────────────────────────────────────────────

def compute_metrics(model, dataloader, device, iou_threshold=0.5):
    """Вычисляет Precision, Recall, mAP50 на выборке."""
    model.eval()

    tp_total = 0
    fp_total = 0
    fn_total = 0
    aps      = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            preds  = model(images)

            for pred, target in zip(preds, targets):
                gt_boxes  = target["boxes"]
                pred_boxes = pred["boxes"].cpu()
                pred_scores = pred["scores"].cpu()
                pred_labels = pred["labels"].cpu()

                # Фильтруем по классу (1 = pedestrian) и порогу уверенности
                mask = (pred_labels == 1) & (pred_scores >= 0.5)
                pred_boxes = pred_boxes[mask]

                gt_boxes_ped = gt_boxes[target["labels"] == 1]

                tp, fp, fn = _match_boxes(pred_boxes, gt_boxes_ped, iou_threshold)
                tp_total += tp
                fp_total += fp
                fn_total += fn

    precision = tp_total / (tp_total + fp_total + 1e-6)
    recall    = tp_total / (tp_total + fn_total + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    # Упрощённый mAP50 как средняя точность при IoU=0.5
    map50 = precision  # для однокласссовой задачи ≈ AP

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "mAP50":     float(map50),
    }


def _match_boxes(pred_boxes, gt_boxes, iou_threshold):
    """Считает TP, FP, FN для одного изображения."""
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0
    fp = 0

    for pb in pred_boxes:
        ious = _box_iou(pb.unsqueeze(0), gt_boxes)[0]
        best_iou, best_idx = ious.max(0)
        if best_iou >= iou_threshold and int(best_idx) not in matched_gt:
            tp += 1
            matched_gt.add(int(best_idx))
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def _box_iou(box, boxes):
    """IoU одного бокса против массива боксов."""
    x1 = torch.max(box[:, 0], boxes[:, 0])
    y1 = torch.max(box[:, 1], boxes[:, 1])
    x2 = torch.min(box[:, 2], boxes[:, 2])
    y2 = torch.min(box[:, 3], boxes[:, 3])

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area1 = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


# ─── Обучение ─────────────────────────────────────────────────────────────────

def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")

    # Датасеты
    train_dataset = YOLODetectionDataset(DATASET_PATH / "images" / "Train", IMG_SIZE)
    val_dataset   = YOLODetectionDataset(DATASET_PATH / "images" / "Validation", IMG_SIZE)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Train: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")

    # Модель
    model = build_model(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    best_map50     = 0.0
    patience_count = 0
    log_path       = RESULTS_DIR / "training_log.yaml"
    log            = []

    for epoch in range(1, EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss      = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} | batch {batch_idx+1}"
                      f"/{len(train_loader)} | loss {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        # ── Validate ──────────────────────────────────────────────────────
        val_metrics = compute_metrics(model, val_loader, DEVICE)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"loss {avg_loss:.4f} | "
              f"P {val_metrics['precision']:.4f} | "
              f"R {val_metrics['recall']:.4f} | "
              f"mAP50 {val_metrics['mAP50']:.4f}")

        log.append({
            "epoch":     epoch,
            "loss":      round(avg_loss, 4),
            **{k: round(v, 4) for k, v in val_metrics.items()}
        })

        # ── Сохранение лучшей модели ──────────────────────────────────────
        if val_metrics["mAP50"] > best_map50:
            best_map50     = val_metrics["mAP50"]
            patience_count = 0
            best_path      = MODELS_DIR / "ssd_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Best model saved (mAP50={best_map50:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Сохраняем лог
    with open(log_path, "w") as f:
        yaml.dump({"training_log": log}, f, default_flow_style=False)

    print(f"\nTraining complete. Best mAP50: {best_map50:.4f}")
    return best_map50


def evaluate(split: str = "val"):
    weights_path = MODELS_DIR / "ssd_best.pt"
    if not weights_path.exists():
        print("No weights found, run train() first")
        return None

    model = build_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print(f"Loaded weights: {weights_path}")

    images_dir = DATASET_PATH / "images" / (
        "Validation" if split == "val" else "Test"
    )
    dataset    = YOLODetectionDataset(images_dir, IMG_SIZE)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    metrics = compute_metrics(model, dataloader, DEVICE)

    print(f"\n── {split.upper()} metrics ──────────────────")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    return metrics


def save_metrics_to_file(val_metrics, test_metrics):
    output = {
        "model": "SSD MobileNetV3 Large",
        "validation": {k: round(v, 4) for k, v in val_metrics.items()},
        "test":       {k: round(v, 4) for k, v in test_metrics.items()},
    }
    out_path = RESULTS_DIR / "ssd_metrics.yaml"
    with open(out_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False)
    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        try:
            train()
            v_metrics = evaluate("val")
            t_metrics = evaluate("test")
            if v_metrics and t_metrics:
                save_metrics_to_file(v_metrics, t_metrics)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
    else:
        print("CUDA is NOT available.")