import rasterio
import numpy as np
from sklearn.preprocessing import StandardScaler

from patchify import patchify
import os
import json # New import for saving scaler params

# --- Configuration ---
FILE_PATH = 'FULL_ALMORA_SUSCEPTIBILITY_STACK_10_01_2026.tif'
TILE_SIZE = 256 
STEP_SIZE = 128 
SAVE_DIR = "training_data"

# --- 1. Load Data and Extract Metadata ---
try:
    with rasterio.open(FILE_PATH) as src:
        full_stack = src.read()
        profile = src.profile
        H, W = src.height, src.width
        N_CHANNELS = src.count
except rasterio.RasterioIOError:
    print(f"Error: Could not open file at {FILE_PATH}. Please ensure the file is accessible.")
    exit()

print(f"File loaded successfully. Dimensions: (Bands: {N_CHANNELS}, Height: {H}, Width: {W})")

# --- 2. Separate Predictors (X) and Target (Y) ---
X_predictors = full_stack[:8, :, :]  # 8 Predictor Bands
Y_mask = full_stack[8, :, :]          # 1 Target Band (Fire_Mask)

# --- 3. Reshape for Deep Learning Frameworks (H, W, C) ---
X_data = np.transpose(X_predictors, (1, 2, 0)) 
Y_data = Y_mask[..., np.newaxis]

# --- 4. Normalization (Standard Scaling - Z-score) ---
X_normalized = np.zeros_like(X_data, dtype=np.float32)
scalers = {} # To store the scalers/parameters

for i in range(X_data.shape[-1]): 
    band_data = X_data[:, :, i].reshape(-1, 1) 
    
    scaler = StandardScaler()
    scaler.fit(band_data)
    
    X_normalized[:, :, i] = scaler.transform(band_data).reshape(H, W)
    
    # Store parameters
    scalers[f'band_{i+1}'] = {'mean': scaler.mean_[0].astype(float), 'std': scaler.scale_[0].astype(float)}

print("\nFeature Normalization Complete.")

# --- 5. Tiling/Patching ---
H_crop = H - (H % TILE_SIZE)
W_crop = W - (W % TILE_SIZE)

X_cropped = X_normalized[:H_crop, :W_crop, :]
Y_cropped = Y_data[:H_crop, :W_crop, :]

X_patches = patchify(X_cropped, (TILE_SIZE, TILE_SIZE, X_cropped.shape[-1]), step=STEP_SIZE)
Y_patches = patchify(Y_cropped, (TILE_SIZE, TILE_SIZE, Y_cropped.shape[-1]), step=STEP_SIZE)

X_train_samples = X_patches.reshape(-1, TILE_SIZE, TILE_SIZE, X_cropped.shape[-1])
Y_train_samples = Y_patches.reshape(-1, TILE_SIZE, TILE_SIZE, Y_cropped.shape[-1])

# --- 6. Spatial Train / Validation Split (NO RANDOM SPLIT) ---

# Get tile grid shape
n_tiles_y = X_patches.shape[0]  # rows of tiles
n_tiles_x = X_patches.shape[1]  # cols of tiles

# Decide split point (e.g., top 80% for training, bottom 20% for validation)
split_row = int(0.8 * n_tiles_y)

# Split patches spatially (north-south split)
X_train = X_patches[:split_row, :, :, :, :].reshape(-1, TILE_SIZE, TILE_SIZE, X_cropped.shape[-1])
Y_train = Y_patches[:split_row, :, :, :, :].reshape(-1, TILE_SIZE, TILE_SIZE, Y_cropped.shape[-1])

X_val = X_patches[split_row:, :, :, :, :].reshape(-1, TILE_SIZE, TILE_SIZE, X_cropped.shape[-1])
Y_val = Y_patches[split_row:, :, :, :, :].reshape(-1, TILE_SIZE, TILE_SIZE, Y_cropped.shape[-1])

# --- 6B. REMOVE EMPTY / NEAR-EMPTY FIRE PATCHES (TRAINING ONLY) ---

# Count number of fire pixels per training patch
fire_pixel_count = np.sum(Y_train, axis=(1, 2, 3))

print("Before filtering:")
print("  Total training patches:", len(Y_train))
print("  Mean fire pixels per patch:", fire_pixel_count.mean())

# Threshold: minimum fire pixels required in a patch
MIN_FIRE_PIXELS = 20   # 🔥 safe value for 256x256 patches

keep_idx = fire_pixel_count >= MIN_FIRE_PIXELS

X_train = X_train[keep_idx]
Y_train = Y_train[keep_idx]

print("After filtering:")
print("  Remaining training patches:", len(Y_train))
print("  Mean fire pixels per patch:", np.sum(Y_train, axis=(1,2,3)).mean())

print(f"Training Samples (spatial): {X_train.shape[0]}")
print(f"Validation Samples (spatial): {X_val.shape[0]}")
print("Train fire pixel ratio:", np.mean(Y_train))
print("Val fire pixel ratio:", np.mean(Y_val))


# --- 7. Save Arrays and Scaler Params to Disk ---
os.makedirs(SAVE_DIR, exist_ok=True)

np.save(os.path.join(SAVE_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(SAVE_DIR, 'Y_train.npy'), Y_train)
np.save(os.path.join(SAVE_DIR, 'X_val.npy'), X_val)
np.save(os.path.join(SAVE_DIR, 'Y_val.npy'), Y_val)

# Save the scaler parameters (CRUCIAL for classification script)
SCALER_FILE = os.path.join(SAVE_DIR, 'scaler_params.json')
with open(SCALER_FILE, 'w') as f:
    json.dump(scalers, f)

print(f"\nTraining data successfully saved to the '{SAVE_DIR}' folder.")
print(f"Scaler parameters successfully saved to {SCALER_FILE}")