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

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
torch.backends.cudnn.enabled = False 
torch.backends.cuda.matmul.allow_tf32 = False

SCRIPT_PATH = Path(__file__).resolve()
BASE_PATH   = SCRIPT_PATH.parent.parent
DATASET_DIR = BASE_PATH / "dataset"
DATA_YAML   = DATASET_DIR / "data.yaml"
OUTPUT_DIR  = SCRIPT_PATH.parent / "comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_CONFIG = [
    {"name": "yolov12n", "path": BASE_PATH / "neural_networks/yolov12/weights/yolov12_best.pt", "type": "ultralytics"},
    {"name": "RT-DETR-L", "path": BASE_PATH / "neural_networks/rf_detr_nano/weights/rtdetr_best.pt", "type": "ultralytics"},
    {"name": "SSD_MobV3", "path": BASE_PATH / "neural_networks/ssd_mobilenetv3/weights/ssd_best.pt", "type": "ssd"}
]

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

def calculate_ssd_metrics(pred_boxes, gt_boxes, iou_thresholds):
    results = []
    for iou_th in iou_thresholds:
        if len(gt_boxes) == 0:
            results.append({'tp': 0, 'fp': len(pred_boxes), 'fn': 0})
            continue
        if len(pred_boxes) == 0:
            results.append({'tp': 0, 'fp': 0, 'fn': len(gt_boxes)})
            continue
            
        matched_gt = set()
        tp = 0
        for pb in pred_boxes:
            ious = ssd_box_iou(pb.unsqueeze(0), gt_boxes)
            best_iou, best_idx = ious.max(0)
            if best_iou >= iou_th and int(best_idx) not in matched_gt:
                tp += 1
                matched_gt.add(int(best_idx))
        results.append({'tp': tp, 'fp': len(pred_boxes) - tp, 'fn': len(gt_boxes) - len(matched_gt)})
    return results

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
    print(f"   - Валидация {cfg['name']} ({subset_name})...")
    model = RTDETR(str(cfg['path'])) if "rtdetr" in cfg['name'].lower() else YOLO(str(cfg['path']))
    
    tmp_txt = DATASET_DIR / f"tmp_list_{subset_name}.txt"
    with open(tmp_txt, 'w', encoding='utf-8') as f: f.write('\n'.join(file_list))
    
    with open(DATA_YAML) as f: y_cfg = yaml.safe_load(f)
    y_cfg.update({'path': '', 'test': str(tmp_txt.resolve())})
    tmp_yaml = DATASET_DIR / f"tmp_yaml_{subset_name}.yaml"
    with open(tmp_yaml, 'w', encoding='utf-8') as f: yaml.dump(y_cfg, f)

    results = model.val(data=str(tmp_yaml), split='test', imgsz=640, batch=1, device=0, verbose=False, plots=False)
    os.remove(tmp_txt); os.remove(tmp_yaml)
    
    return {
        "Precision": round(results.box.p.mean(), 4),
        "Recall": round(results.box.r.mean(), 4),
        "mAP50": round(results.box.map50, 4),
        "mAP50_95": round(results.box.map, 4),
        "Lat_ms": round(results.speed['inference'], 2)
    }

def validate_ssd(cfg, file_list, subset_name):
    print(f"   - Валидация {cfg['name']} ({subset_name})...")
    model = build_ssd_model().to("cuda")
    model.load_state_dict(torch.load(cfg['path'], map_location="cuda"))
    model.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    iou_ths = np.linspace(0.5, 0.95, 10) # Пороги для mAP50-95
    stats = {th: {'tp': 0, 'fp': 0, 'fn': 0} for th in iou_ths}
    latencies = []

    with torch.no_grad():
        for p in file_list:
            img_pil = Image.open(p).convert("RGB").resize((320, 320))
            img_t = transform(img_pil).unsqueeze(0).to("cuda")
            
            t1 = time.perf_counter()
            preds = model(img_t)
            latencies.append((time.perf_counter() - t1) * 1000)

            label_p = Path(p.replace("images", "labels")).with_suffix(".txt")
            gt_boxes = []
            with open(label_p) as f:
                for line in f:
                    _, cx, cy, bw, bh = map(float, line.split())
                    gt_boxes.append([(cx-bw/2)*320, (cy-bh/2)*320, (cx+bw/2)*320, (cy+bh/2)*320])
            
            gt_boxes = torch.tensor(gt_boxes)
            p_boxes = preds[0]["boxes"].cpu()[preds[0]["scores"].cpu() >= 0.5]
            
            frame_stats = calculate_ssd_metrics(p_boxes, gt_boxes, iou_ths)
            for i, th in enumerate(iou_ths):
                stats[th]['tp'] += frame_stats[i]['tp']
                stats[th]['fp'] += frame_stats[i]['fp']
                stats[th]['fn'] += frame_stats[i]['fn']

    aps = []
    for th in iou_ths:
        p = stats[th]['tp'] / (stats[th]['tp'] + stats[th]['fp'] + 1e-6)
        r = stats[th]['tp'] / (stats[th]['tp'] + stats[th]['fn'] + 1e-6)
        aps.append(p) 
    
    return {
        "Precision": round(aps[0], 4),
        "Recall": round(stats[0.5]['tp'] / (stats[0.5]['tp'] + stats[0.5]['fn'] + 1e-6), 4),
        "mAP50": round(aps[0], 4),
        "mAP50_95": round(np.mean(aps), 4),
        "Lat_ms": round(np.mean(latencies), 2)
    }

def plot_accuracy_matrix(report):
    metrics = ["Precision", "Recall", "mAP50", "mAP50_95"]
    models = [r['model'] for r in report]
    x = np.arange(len(models))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Сравнение метрик точности: День vs Ночь', fontsize=18, fontweight='bold')
    
    axes = axes.flatten()
    colors = {'day': '#f1c40f', 'night': '#34495e'}

    for i, m in enumerate(metrics):
        day_vals = [r['day'][m] for r in report]
        night_vals = [r['night'][m] for r in report]
        
        axes[i].bar(x - width/2, day_vals, width, label='День', color=colors['day'], edgecolor='black', alpha=0.8)
        axes[i].bar(x + width/2, night_vals, width, label='Ночь', color=colors['night'], edgecolor='black', alpha=0.9)
        
        axes[i].set_title(f'Метрика: {m}', fontsize=14, fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models)
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)
        if i == 0: axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "accuracy.png", dpi=150)

def main():
    day_files, night_files = get_test_subsets()
    report = []

    for cfg in MODELS_CONFIG:
        print(f"\n>>> СТАРТ ТЕСТА: {cfg['name']}")
        if not cfg['path'].exists(): continue

        if cfg['type'] == "ultralytics":
            res_day = validate_ultralytics(cfg, day_files, "day")
            res_night = validate_ultralytics(cfg, night_files, "night")
        else:
            res_day = validate_ssd(cfg, day_files, "day")
            res_night = validate_ssd(cfg, night_files, "night")

        report.append({
            "model": cfg['name'],
            "day": res_day,
            "night": res_night
        })

    with open(OUTPUT_DIR / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    plot_accuracy_matrix(report)

    plt.figure(figsize=(10, 6))
    lats = [r['day']['Lat_ms'] for r in report]
    plt.bar([r['model'] for r in report], lats, color='#e74c3c', edgecolor='black')
    plt.title('Скорость обработки (Inference Time)', fontsize=14, fontweight='bold')
    plt.ylabel('ms (миллисекунды)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(OUTPUT_DIR / "latency.png")

    print(f"\nСравнение завершено. Результаты в папке: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()