import os
import cv2

def flip_yolo_labels(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            cls, x, y, w, h = parts
            new_x = 1.0 - float(x)
            new_lines.append(f"{cls} {new_x:.6f} {y} {w} {h}\n")
    
    with open(output_path, 'w') as f:
        f.writelines(new_lines)

def augment_dataset(root_dir):
    subsets = ['Train', 'Validation', 'Test']
    
    for subset in subsets:
        img_dir = os.path.join(root_dir, 'images', subset)
        lbl_dir = os.path.join(root_dir, 'labels', subset)
        list_file_path = os.path.join(root_dir, f"{subset}.txt")
        
        if not os.path.exists(img_dir):
            continue
            
        print(f"Processing {subset}...")
        
        new_image_paths = []
        
        for img_name in os.listdir(img_dir):
            if img_name.startswith('v') and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                img_path = os.path.join(img_dir, img_name)
                base_name = os.path.splitext(img_name)[0]
                lbl_name = base_name + ".txt"
                lbl_path = os.path.join(lbl_dir, lbl_name)
                
                new_img_name = f"{base_name}_flip{os.path.splitext(img_name)[1]}"
                new_lbl_name = f"{base_name}_flip.txt"
                
                new_img_path = os.path.join(img_dir, new_img_name)
                new_lbl_path = os.path.join(lbl_dir, new_lbl_name)
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                flipped_img = cv2.flip(img, 1) 
                cv2.imwrite(new_img_path, flipped_img)
                
                if os.path.exists(lbl_path):
                    flip_yolo_labels(lbl_path, new_lbl_path)
                else:
                    print(f"Warning: Label not found for {img_name}")
                
                new_image_paths.append(f"images/{subset}/{new_img_name}\n")

        if os.path.exists(list_file_path) and new_image_paths:
            with open(list_file_path, 'a') as f:
                f.writelines(new_image_paths)
            print(f"Added {len(new_image_paths)} flipped images to {subset}.txt")

if __name__ == "__main__":
    dataset_root = "./dataset" 
    augment_dataset(dataset_root)
    print("Done!")