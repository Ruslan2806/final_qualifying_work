import os
import shutil
from pathlib import Path

SCRIPT_PATH   = Path(__file__).resolve()

BASE_PATH     = SCRIPT_PATH.parent.parent.parent 
DATASET_DIR   = BASE_PATH / "dataset"

os.chdir(DATASET_DIR)

import yaml

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

from ultralytics import YOLO

BASE_PATH     = Path(__file__).resolve().parent.parent.parent
DATASET_YAML  = BASE_PATH / "dataset" / "data.yaml"
MODELS_DIR    = Path(__file__).resolve().parent / "weights"
RESULTS_DIR   = Path(__file__).resolve().parent / "results"

MODEL_NAME    = "yolov12n.pt"   
EPOCHS        = 100            
IMG_SIZE      = 640
BATCH_SIZE    = 16             
LEARNING_RATE = 0.01           
DEVICE        = 0              
# ─────────────────────────────────────────────────────────

def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"Starting training on: {DATASET_YAML}")
    results = model.train(
        data    = str(DATASET_YAML),
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        lr0     = LEARNING_RATE,
        device  = DEVICE,
        project = str(RESULTS_DIR),
        name    = "yolov12_pedestrian",
        exist_ok= True,

        # Аугментация
        fliplr  = 0.0,   
        hsv_v   = 0.6,   
        hsv_s   = 0.4,   
        mosaic  = 1.0,   
        
        # Настройки совместимости
        amp     = False, 
        workers = 0,     
        patience= 10,    
    )

    best_weights = RESULTS_DIR / "yolov12_pedestrian" / "weights" / "best.pt"
    if best_weights.exists():
        dest = MODELS_DIR / "yolov12_best.pt"
        shutil.copy(best_weights, dest)
        print(f"Best weights saved to: {dest}")

    return results

def validate():
    weights = MODELS_DIR / "yolov12_best.pt"
    if not weights.exists():
        return

    print(f"Validating: {weights}")
    model   = YOLO(str(weights))
    metrics = model.val(
        data    = str(DATASET_YAML),
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        split   = "val",
        plots   = True,
        project = str(RESULTS_DIR),
        name    = "yolov12_val",
        exist_ok= True,
    )
    return metrics

def test():
    weights = MODELS_DIR / "yolov12_best.pt"
    if not weights.exists():
        return

    print(f"Testing: {weights}")
    model   = YOLO(str(weights))
    metrics = model.val(
        data    = str(DATASET_YAML),
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        device  = DEVICE,
        split   = "test",
        plots   = True,
        project = str(RESULTS_DIR),
        name    = "yolov12_test",
        exist_ok= True,
    )
    return metrics

def save_metrics_to_file(val_metrics, test_metrics):
    output = {
        "model": "YOLOv12",
        "validation": {
            "precision": float(val_metrics.box.p.mean()),
            "recall":    float(val_metrics.box.r.mean()),
            "mAP50":     float(val_metrics.box.map50),
            "mAP50_95":  float(val_metrics.box.map),
        },
        "test": {
            "precision": float(test_metrics.box.p.mean()),
            "recall":    float(test_metrics.box.r.mean()),
            "mAP50":     float(test_metrics.box.map50),
            "mAP50_95":  float(test_metrics.box.map),
        }
    }
    out_path = RESULTS_DIR / "yolov12_metrics.yaml"
    with open(out_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False)
    print(f"\nMetrics saved to: {out_path}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        try:
            train_results = train()
            v_metrics = validate()
            t_metrics = test()

            if v_metrics and t_metrics:
                save_metrics_to_file(v_metrics, t_metrics)
        except Exception as e:
            print(f"An error occurred during training: {e}")
    else:
        print("CUDA is NOT available.")