import os

os.chdir(r"C:\Users\ruslan\Documents\GitHub\final_qualifying_work\dataset")
import shutil
import cv2
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
from tqdm import tqdm

from proc_night_photos import increase_clarity

os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

BASE_PATH         = Path(__file__).resolve().parent.parent.parent
ORIGINAL_DATASET  = BASE_PATH / "dataset"
FIXED_DATASET     = BASE_PATH / "dataset_fixed"

MODELS_DIR    = Path(__file__).parent / "weights"
RESULTS_DIR   = Path(__file__).parent / "results_fixed"

MODEL_NAME    = "yolov8n.pt"   
EPOCHS        = 100            
IMG_SIZE      = 640
BATCH_SIZE    = 16             
DEVICE        = 0              

def prepare_fixed_dataset():
    if FIXED_DATASET.exists():
        print(f"--- Обнаружен готовый датасет: {FIXED_DATASET}. Пропускаем обработку. ---")
        return

    print("--- Начинается предобработка ночных фото (этап 'Fix Night') ---")
    
    for sub in ['images', 'labels']:
        for folder in ['Train', 'Validation', 'Test']:
            (FIXED_DATASET / sub / folder).mkdir(parents=True, exist_ok=True)

    print("Копирование меток...")
    shutil.copytree(ORIGINAL_DATASET / "labels", FIXED_DATASET / "labels", dirs_exist_ok=True)
    
    for folder in ['Train', 'Validation', 'Test']:
        src_img_dir = ORIGINAL_DATASET / "images" / folder
        dst_img_dir = FIXED_DATASET / "images" / folder
        
        files = list(src_img_dir.glob("*.*"))
        for img_path in tqdm(files, desc=f"Processing {folder}"):
            img_name = img_path.name
            if img_name.startswith('v'):
                img = cv2.imread(str(img_path))
                if img is not None:
                    fixed_img = increase_clarity(img)
                    cv2.imwrite(str(dst_img_dir / img_name), fixed_img)
            else:
                shutil.copy(img_path, dst_img_dir / img_name)

    with open(ORIGINAL_DATASET / "data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_config['path'] = str(FIXED_DATASET)
    if 'Train' in data_config: data_config['train'] = data_config.pop('Train')
    if 'Validation' in data_config: data_config['val'] = data_config.pop('Validation')
    if 'Test' in data_config: data_config['test'] = data_config.pop('Test')

    with open(FIXED_DATASET / "data.yaml", 'w') as f:
        yaml.dump(data_config, f)
    
    for txt in ['Train.txt', 'Validation.txt', 'Test.txt']:
        if (ORIGINAL_DATASET / txt).exists():
            shutil.copy(ORIGINAL_DATASET / txt, FIXED_DATASET / txt)

    print(f"--- Предобработка завершена. Новый датасет: {FIXED_DATASET} ---")

def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    os.chdir(str(FIXED_DATASET))
    
    model = YOLO(MODEL_NAME)
    dataset_yaml = FIXED_DATASET / "data.yaml"

    results = model.train(
        data    = str(dataset_yaml),
        epochs  = EPOCHS,
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        lr0     = 0.01,
        device  = DEVICE,
        project = str(RESULTS_DIR),
        name    = "yolov8_fixed_night",
        exist_ok= True,
        amp     = False, 
        workers = 0,
        fliplr  = 0.0,
        hsv_v   = 0.6,
        mosaic  = 1.0
    )

    best_weights = RESULTS_DIR / "yolov8_fixed_night" / "weights" / "best.pt"
    if best_weights.exists():
        dest = MODELS_DIR / "yolov8_fixed_best.pt"
        shutil.copy(best_weights, dest)
        print(f"Best fixed weights saved to: {dest}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        
        prepare_fixed_dataset()
        
        try:
            train()
            print("Обучение на улучшенных фото завершено!")
        except Exception as e:
            print(f"Ошибка: {e}")
    else:
        print("CUDA is NOT available.")