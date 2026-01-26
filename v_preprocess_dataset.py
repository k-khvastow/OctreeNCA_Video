import os
import numpy as np
import cv2
import scipy.io
import natsort
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_ROOT = "/vol/data/BioProject13/data_OCT/OCT"
LABEL_ROOT = "/vol/data/BioProject13/data_OCT/Label/GT_Layers"
OUTPUT_ROOT = "/vol/data/BioProject13/data_OCT/npy_Cropped_400"
TARGET_SIZE = (400, 400) # (Height, Width)

def smart_crop(arr, target_h, target_w, interpolation):
    """
    Center crop or resize if the image is smaller than target.
    """
    h, w = arr.shape[:2]
    
    if w < target_w or h < target_h:
        return cv2.resize(arr, (target_w, target_h), interpolation=interpolation)
    else:
        x1 = int(round((w - target_w) / 2.))
        y1 = int(round((h - target_h) / 2.))
        x_start = max(0, x1)
        y_start = max(0, y1)
        return arr[y_start : y_start + target_h, x_start : x_start + target_w]

def preprocess():
    # We will use a single folder for compressed arrays to keep it tidy, 
    # or you can keep your images/labels structure if you prefer separate files.
    # Here I keep your structure but use compressed numpy files (.npz)
    os.makedirs(os.path.join(OUTPUT_ROOT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels"), exist_ok=True)

    if not os.path.exists(DATA_ROOT):
        raise ValueError(f"Data root not found: {DATA_ROOT}")

    patient_folders = natsort.natsorted(os.listdir(DATA_ROOT))
    
    print(f"Starting preprocessing. Target Crop Size: {TARGET_SIZE}")
    print(f"Output: {OUTPUT_ROOT}")

    for patient_id in tqdm(patient_folders):
        patient_dir = os.path.join(DATA_ROOT, patient_id)
        if not os.path.isdir(patient_dir): continue

        label_path = os.path.join(LABEL_ROOT, f"{patient_id}.mat")
        if not os.path.exists(label_path):
            continue
            
        try:
            mat = scipy.io.loadmat(label_path)
            if 'Layer' not in mat: continue
            layers_data = mat['Layer'] 
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
            continue

        images = natsort.natsorted([x for x in os.listdir(patient_dir) if x.endswith('.bmp')])
        
        for frame_idx, img_name in enumerate(images):
            img_path = os.path.join(patient_dir, img_name)
            
            # 1. Load as standard uint8 (0-255)
            # DO NOT normalize to float32 here. Do it in your DataLoader.
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue

            orig_h, orig_w = img.shape

            if frame_idx >= layers_data.shape[1]: continue
            
            # 2. Generate Mask
            # We can use int8/uint8 immediately for the mask
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            y_grid = np.arange(orig_h).reshape(orig_h, 1)
            
            num_layers = layers_data.shape[0]
            for l_idx in range(num_layers):
                layer_surface = layers_data[l_idx, frame_idx, :] 
                if layer_surface.shape[0] != orig_w:
                    layer_surface = cv2.resize(layer_surface[None, :], (orig_w, 1), interpolation=cv2.INTER_LINEAR)[0]
                
                # Add 1 where y > surface
                mask += (y_grid > layer_surface.reshape(1, orig_w)).astype(np.uint8)

            mask[mask == num_layers] = 0
            
            # 3. Apply CROP
            img_cropped = smart_crop(img, TARGET_SIZE[0], TARGET_SIZE[1], cv2.INTER_LINEAR)
            mask_cropped = smart_crop(mask, TARGET_SIZE[0], TARGET_SIZE[1], cv2.INTER_NEAREST)

            # 4. Save as Compressed Numpy (.npz)
            # This saves drastically more space than .npy
            out_name = f"{patient_id}_{frame_idx}.npz"
            
            # Save Image: uint8, shape (H, W, 1)
            np.savez_compressed(
                os.path.join(OUTPUT_ROOT, "images", out_name), 
                data=img_cropped[..., None]
            )
            
            # Save Label: uint8, shape (H, W)
            np.savez_compressed(
                os.path.join(OUTPUT_ROOT, "labels", out_name), 
                data=mask_cropped
            )

if __name__ == "__main__":
    preprocess()