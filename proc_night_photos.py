import cv2
import numpy as np
import os
from tqdm import tqdm

def get_gamma_lut(gamma=1.8):
    inv_gamma = 1.0 / gamma
    return np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")

GAMMA_LUT = get_gamma_lut(1.8)
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

def increase_clarity(image):
    
    img = cv2.LUT(image, GAMMA_LUT)
    
    img = cv2.bilateralFilter(img, 5, 65, 65)
    
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = CLAHE.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    
    gaussian = cv2.GaussianBlur(img, (0, 0), 1.0)
    img = cv2.addWeighted(img, 1.6, gaussian, -0.6, 0)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 0.8) 
    img = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    
    return img

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(extensions)]

    if not files:
        print(f"В папке {input_dir} фото не найдены.")
        return

    for filename in tqdm(files, desc="Обработка"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path)
        if image is None:
            continue

        processed_image = increase_clarity(image)

        cv2.imwrite(output_path, processed_image)

    print(f"\nГотово! Обработанные фото сохранены в: {output_dir}")

if __name__ == "__main__":
    SOURCE_FOLDER = ""   
    RESULT_FOLDER = "" 
    
    process_directory(SOURCE_FOLDER, RESULT_FOLDER)