import os
import json
import time
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from functools import partial
from ultralytics import YOLO, RTDETR

# --- 1. НАСТРОЙКИ GPU ДЛЯ RTX 5050 (Blackwell) ---
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
torch.backends.cudnn.enabled = False 
torch.backends.cuda.matmul.allow_tf32 = False

# --- 2. ПУТИ ---
SCRIPT_PATH = Path(__file__).resolve()
BASE_PATH   = SCRIPT_PATH.parent.parent
DATASET_DIR = BASE_PATH / "dataset"
DATA_YAML   = DATASET_DIR / "data.yaml"
OUTPUT_DIR  = SCRIPT_PATH.parent / "comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_CONFIG = [
    {"name": "YOLOv8n", "path": BASE_PATH / "neural_networks/yolov8/weights/yolov8_best.pt", "type": "ultralytics"},
    {"name": "RT-DETR-L", "path": BASE_PATH / "neural_networks/rf_detr_nano/weights/rtdetr_best.pt", "type": "ultralytics"},
    {"name": "SSD_MobV3", "path": BASE_PATH / "neural_networks/ssd_mobilenetv2/weights/ssd_best.pt", "type": "ssd"}
]

# =====================================================================
# ЛОГИКА SSD (ПОЛНОЕ ВОССОЗДАНИЕ ИЗ SSD_MOBILENETV2.PY)
# =====================================================================

def build_ssd_model(num_classes=2):
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
    in_channels = [672, 480, 512, 256, 256, 128]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels=in_channels, num_anchors=num_anchors, 
        num_classes=num_classes, norm_layer=partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)
    )
    return model

def ssd_box_iou(box, boxes):
    inter = (torch.min(box[:, 2], boxes[:, 2]) - torch.max(box[:, 0], boxes[:, 0])).clamp(0) * \
            (torch.min(box[:, 3], boxes[:, 3]) - torch.max(box[:, 1], boxes[:, 1])).clamp(0)
    area1 = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)

def ssd_match_boxes(pred_boxes, gt_boxes):
    if len(gt_boxes) == 0: return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0: return 0, 0, len(gt_boxes)
    matched_gt = set()
    tp, fp = 0, 0
    for pb in pred_boxes:
        ious = ssd_box_iou(pb.unsqueeze(0), gt_boxes)
        best_iou, best_idx = ious.max(0)
        if best_iou >= 0.5 and int(best_idx) not in matched_gt:
            tp += 1
            matched_gt.add(int(best_idx))
        else: fp += 1
    return tp, fp, len(gt_boxes) - len(matched_gt)

# =====================================================================
# ТЕСТОВЫЕ ДВИЖКИ
# =====================================================================

def get_test_subsets():
    img_dir = DATASET_DIR / "images" / "Test"
    day, night = [], []
    for f in img_dir.iterdir():
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg'):
            if (DATASET_DIR / "labels/Test" / (f.stem + ".txt")).exists():
                if f.name.startswith('v'): night.append(str(f.resolve()))
                else: day.append(str(f.resolve()))
    return day, night

def validate_ultralytics(cfg, file_list, subset_name):
    print(f"   - Валидация {subset_name}...")
    model = RTDETR(str(cfg['path'])) if "rtdetr" in cfg['name'].lower() else YOLO(str(cfg['path']))
    
    tmp_txt = DATASET_DIR / f"tmp_list_{subset_name}.txt"
    with open(tmp_txt, 'w', encoding='utf-8') as f: f.write('\n'.join(file_list))
    
    with open(DATA_YAML) as f: y_cfg = yaml.safe_load(f)
    y_cfg.update({'path': '', 'test': str(tmp_txt.resolve())})
    tmp_yaml = DATASET_DIR / f"tmp_yaml_{subset_name}.yaml"
    with open(tmp_yaml, 'w') as f: yaml.dump(y_cfg, f)

    results = model.val(data=str(tmp_yaml), split='test', imgsz=640, batch=1, device=0, verbose=False, plots=False)
    
    os.remove(tmp_txt); os.remove(tmp_yaml)
    return {
        "mAP50": round(results.box.map50, 4),
        "Recall": round(results.box.r.mean(), 4),
        "Lat_ms": round(results.speed['inference'], 2)
    }

def validate_ssd(cfg, file_list, subset_name):
    print(f"   - Валидация {subset_name}...")
    model = build_ssd_model().to("cuda")
    model.load_state_dict(torch.load(cfg['path'], map_location="cuda"))
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    tp_t, fp_t, fn_t = 0, 0, 0
    latencies = []

    with torch.no_grad():
        for p in file_list:
            img_pil = Image.open(p).convert("RGB").resize((320, 320))
            img_t = transform(img_pil).unsqueeze(0).to("cuda")
            
            # Замер времени
            t1 = time.perf_counter()
            preds = model(img_t)
            latencies.append((time.perf_counter() - t1) * 1000)

            # Расчет метрик (как в вашем скрипте)
            label_p = Path(p.replace("images", "labels")).with_suffix(".txt")
            gt_boxes = []
            with open(label_p) as f:
                for line in f:
                    _, cx, cy, bw, bh = map(float, line.split())
                    gt_boxes.append([(cx-bw/2)*320, (cy-bh/2)*320, (cx+bw/2)*320, (cy+bh/2)*320])
            
            gt_boxes = torch.tensor(gt_boxes)
            p_boxes = preds[0]["boxes"].cpu()[preds[0]["scores"].cpu() >= 0.5]
            
            tp, fp, fn = ssd_match_boxes(p_boxes, gt_boxes)
            tp_t += tp; fp_t += fp; fn_t += fn

    p = tp_t / (tp_t + fp_t + 1e-6)
    r = tp_t / (tp_t + fn_t + 1e-6)
    return {"mAP50": round(p, 4), "Recall": round(r, 4), "Lat_ms": round(np.mean(latencies), 2)}

# =====================================================================
# ОСНОВНОЙ ЦИКЛ
# =====================================================================

def main():
    day_files, night_files = get_test_subsets()
    all_files = day_files + night_files
    report = []

    for cfg in MODELS_CONFIG:
        print(f"\n>>> СРАВНЕНИЕ МОДЕЛИ: {cfg['name']}")
        if not cfg['path'].exists():
            print(f"!!! Файл {cfg['path'].name} не найден. Пропуск.")
            continue

        if cfg['type'] == "ultralytics":
            res_all   = validate_ultralytics(cfg, all_files, "all")
            res_day   = validate_ultralytics(cfg, day_files, "day")
            res_night = validate_ultralytics(cfg, night_files, "night")
        else:
            res_all   = validate_ssd(cfg, all_files, "all")
            res_day   = validate_ssd(cfg, day_files, "day")
            res_night = validate_ssd(cfg, night_files, "night")

        report.append({
            "model": cfg['name'],
            "all": res_all, "day": res_day, "night": res_night
        })

    # Сохранение JSON
    with open(OUTPUT_DIR / "comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # ПОСТРОЕНИЕ ГРАФИКОВ
    names = [r['model'] for r in report]
    m_day = [r['day']['mAP50'] for r in report]
    m_night = [r['night']['mAP50'] for r in report]
    lats = [r['all']['Lat_ms'] for r in report]

    x = np.arange(len(names))
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, m_day, 0.4, label='День (mAP50)', color='#f1c40f')
    plt.bar(x + 0.2, m_night, 0.4, label='Ночь (mAP50)', color='#34495e')
    plt.xticks(x, names); plt.ylabel('Точность'); plt.legend(); plt.grid(axis='y', alpha=0.3)
    plt.title('Сравнение точности детектирования'); plt.savefig(OUTPUT_DIR / "accuracy.png")

    plt.figure(figsize=(10, 6))
    plt.bar(names, lats, color='#e74c3c')
    plt.ylabel('Задержка (мс)'); plt.title('Среднее время инференса (batch=1)'); plt.savefig(OUTPUT_DIR / "latency.png")

    print(f"\nГотово! Отчет и графики сохранены в: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()