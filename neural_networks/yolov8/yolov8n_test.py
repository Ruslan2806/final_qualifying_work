import os
import shutil
from pathlib import Path
import yaml
import torch

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import torch
torch.backends.cudnn.enabled = False 
torch.backends.cuda.matmul.allow_tf32 = False

from ultralytics import YOLO

SCRIPT_PATH = Path(__file__).resolve()
BASE_PATH   = SCRIPT_PATH.parent.parent.parent
DATASET_DIR = BASE_PATH / "dataset"
DATA_YAML   = DATASET_DIR / "data.yaml"
MODEL_PATH  = SCRIPT_PATH.parent / "weights" / "yolov8_best.pt"

def run_split_test():
    for cache_file in DATASET_DIR.glob("**/*.cache"):
        try:
            os.remove(cache_file)
            print(f"Удален старый кэш: {cache_file.name}")
        except: pass

    if not MODEL_PATH.exists():
        print(f"ОШИБКА: Модель не найдена: {MODEL_PATH}")
        return

    model = YOLO(str(MODEL_PATH))
    test_images_dir = DATASET_DIR / "images" / "Test"
    
    if not test_images_dir.exists():
        print(f"ОШИБКА: Папка {test_images_dir} не найдена!")
        return

    valid_ext = ('.png', '.jpg', '.jpeg')
    day_files = []
    night_files = []

    for img_file in test_images_dir.iterdir():
        if img_file.suffix.lower() in valid_ext:
            label_file = DATASET_DIR / "labels" / "Test" / (img_file.stem + ".txt")
            
            if not label_file.exists():
                continue
                
            abs_path = str(img_file.resolve())
            if img_file.name.startswith('v'):
                night_files.append(abs_path)
            else:
                day_files.append(abs_path)

    print(f"Валидных пар (фото+метка) найдено: День={len(day_files)}, Ночь={len(night_files)}")

    def get_metrics(file_list, name):
        if not file_list: return None
        
        tmp_txt = DATASET_DIR / f"tmp_test_list_{name}.txt"
        with open(tmp_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(file_list))
        
        with open(DATA_YAML, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config['path'] = '' 
        config['test'] = str(tmp_txt.resolve())
        
        tmp_yaml = DATASET_DIR / f"tmp_config_{name}.yaml"
        with open(tmp_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        results = model.val(
            data=str(tmp_yaml),
            split='test',
            imgsz=640,
            batch=16,
            device=0,
            amp=False,
            plots=False,
            verbose=False
        )

        if tmp_txt.exists(): os.remove(tmp_txt)
        if tmp_yaml.exists(): os.remove(tmp_yaml)
        
        return {
            "Precision": results.box.p.mean(),
            "Recall": results.box.r.mean(),
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map
        }

    stats = {}
    stats["День"] = get_metrics(day_files, "day")
    stats["Ночь"] = get_metrics(night_files, "night")

    print("\n" + "="*55)
    print(f"{'Метрика':<15} | {'День (Solar)':<15} | {'Ночь (Night)':<15}")
    print("-" * 55)
    for m in ["Precision", "Recall", "mAP50", "mAP50-95"]:
        d_val = f"{stats['День'][m]:.4f}" if stats["День"] else "N/A"
        n_val = f"{stats['Ночь'][m]:.4f}" if stats["Ночь"] else "N/A"
        print(f"{m:<15} | {d_val:<15} | {n_val:<15}")
    print("="*55)

if __name__ == "__main__":
    if torch.cuda.is_available():
        run_split_test()