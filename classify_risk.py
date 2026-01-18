import rasterio
import numpy as np
import os
os.environ["TF_DATA_AUTOTUNE_RAM_LIMIT"] = "268435456"  # 256 MB

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanIoU

import json
from osgeo import gdal, osr
from PIL import Image

# --- CONFIGURATION ---
MODEL_PATH = 'unet_fire_susceptibility_model.h5'
DATA_DIR = "training_data"
SCALER_PARAMS_PATH = os.path.join(DATA_DIR, 'scaler_params.json')

TILE_SIZE = 256
INFER_STEP = 192   # 🔥 NEW: overlap stride (fixes striping)

# --- GDAL Color Map ---
RISK_COLOR_MAP = {
    1: (0, 150, 0, 255),     # Low Risk
    2: (255, 165, 0, 255),   # Medium Risk
    3: (255, 0, 0, 255),     # High Risk
    0: (0, 0, 0, 0)
}

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def embed_color_map(filepath, color_map):
    ds = gdal.Open(filepath, gdal.GA_Update)
    if ds is None:
        print(f"Error: Could not open {filepath} with GDAL.")
        return

    band = ds.GetRasterBand(1)
    color_table = gdal.ColorTable()
    for value, color in color_map.items():
        color_table.SetColorEntry(value, color)

    band.SetRasterColorTable(color_table)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    band.FlushCache()
    ds = None

def mean_io_u(y_true, y_pred):
    return MeanIoU(num_classes=2)(y_true, tf.round(y_pred))

def generate_png_preview(risk_map_array, color_map, output_path):
    H, W = risk_map_array.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

    for code, (r, g, b, a) in color_map.items():
        if code > 0:
            mask = risk_map_array == code
            rgb_image[mask, 0] = r
            rgb_image[mask, 1] = g
            rgb_image[mask, 2] = b

    Image.fromarray(rgb_image, 'RGB').save(output_path)
    print(f"PNG preview generated at: {output_path}")
    return output_path

# ------------------------------------------------------------
# MAIN CLASSIFICATION FUNCTION
# ------------------------------------------------------------
def run_unet_classification(input_geotiff_path: str,
                            output_geotiff_path: str,
                            output_png_path: str):

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PARAMS_PATH):
        raise FileNotFoundError(f"Scaler params not found: {SCALER_PARAMS_PATH}")

    # Load model and scaler
    model = load_model(
        MODEL_PATH,
        custom_objects={'mean_io_u': mean_io_u},
        compile=False
    )

    with open(SCALER_PARAMS_PATH, 'r') as f:
        SCALER_PARAMS = json.load(f)

    # Load input GeoTIFF
    with rasterio.open(input_geotiff_path) as src:
        full_stack = src.read()
        profile = src.profile
        H, W = src.height, src.width

    # --------------------------------------------------------
    # Normalization
    # --------------------------------------------------------
    X_predictors = full_stack[:8, :, :]
    X_data = np.transpose(X_predictors, (1, 2, 0)).astype(np.float32)
    X_normalized = np.zeros_like(X_data, dtype=np.float32)

    for i in range(X_data.shape[-1]):
        mean = SCALER_PARAMS[f'band_{i+1}']['mean']
        std = SCALER_PARAMS[f'band_{i+1}']['std']
        std = std if std != 0 else 1.0
        X_normalized[:, :, i] = (X_data[:, :, i] - mean) / std
    
    X_normalized = np.clip(X_normalized, -2.5, 2.5)

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------
    pad_h = TILE_SIZE - H if H < TILE_SIZE else (TILE_SIZE - H % TILE_SIZE if H % TILE_SIZE != 0 else 0)
    pad_w = TILE_SIZE - W if W < TILE_SIZE else (TILE_SIZE - W % TILE_SIZE if W % TILE_SIZE != 0 else 0)


    X_padded = np.pad(
        X_normalized,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode='reflect'
    )

    H_pad, W_pad, _ = X_padded.shape

# --------------------------------------------------------
# PROFESSIONAL FIX: Gaussian-weighted overlapping inference
# --------------------------------------------------------

    INFER_STEP = 128
    BATCH_SIZE = 4   # safe for CPU, increase if GPU available

    def gaussian_window(size):
        ax = np.linspace(-(size // 2), size // 2, size)
        xx, yy = np.meshgrid(ax, ax)
        sigma = size / 6.0
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel / kernel.max()
    weight_kernel = gaussian_window(TILE_SIZE)[..., np.newaxis]

    probability_map_padded = np.zeros((H_pad, W_pad, 1), dtype=np.float32)
    weight_map = np.zeros((H_pad, W_pad, 1), dtype=np.float32)

    tiles = []
    positions = []

    for i in range(0, H_pad - TILE_SIZE + 1, INFER_STEP):
        for j in range(0, W_pad - TILE_SIZE + 1, INFER_STEP):
            tile = X_padded[i:i + TILE_SIZE, j:j + TILE_SIZE, :]
            tiles.append(tile)
            positions.append((i, j))
        if len(tiles) == BATCH_SIZE:
            preds = model.predict(np.array(tiles), verbose=0)
            for (ii, jj), pred in zip(positions, preds):
                probability_map_padded[ii:ii + TILE_SIZE, jj:jj + TILE_SIZE, :] += pred * weight_kernel
                weight_map[ii:ii + TILE_SIZE, jj:jj + TILE_SIZE, :] += weight_kernel
            tiles, positions = [], []

# Process remaining tiles
    if tiles:
        preds = model.predict(np.array(tiles), verbose=0)
        for (ii, jj), pred in zip(positions, preds):
            probability_map_padded[ii:ii + TILE_SIZE, jj:jj + TILE_SIZE, :] += pred * weight_kernel
            weight_map[ii:ii + TILE_SIZE, jj:jj + TILE_SIZE, :] += weight_kernel
    probability_map_padded /= np.maximum(weight_map, 1e-6)
    probability_map = probability_map_padded[:H, :W, :]


    # --- SAFETY FALLBACK: ensure at least one prediction ---
    if np.sum(weight_map) == 0:
        print("⚠️ No overlapping tiles processed, running single full-tile inference.")

        tile = X_padded[:TILE_SIZE, :TILE_SIZE, :]
        tile = np.expand_dims(tile, axis=0)

        pred_tile = model.predict(tile, verbose=0)[0]

        probability_map_padded[:TILE_SIZE, :TILE_SIZE, :] = pred_tile
        weight_map[:TILE_SIZE, :TILE_SIZE, :] = 1.0


    print("DEBUG:")
    print("  Max probability:", probability_map_padded.max())
    print("  Mean probability:", probability_map_padded.mean())

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------
    T_high = np.percentile(probability_map, 99.5)
    T_low  = np.percentile(probability_map, 98.0)


    risk_map = np.ones_like(probability_map, dtype=np.uint8)
    risk_map[probability_map >= T_high] = 3
    risk_map[(probability_map >= T_low) & (probability_map < T_high)] = 2

    risk_map_final = risk_map.squeeze()

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    profile.update(dtype=rasterio.uint8, count=1, photometric='MINISBLACK')

    with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
        dst.write(risk_map_final, 1)

    embed_color_map(output_geotiff_path, RISK_COLOR_MAP)
    generate_png_preview(risk_map_final, RISK_COLOR_MAP, output_png_path)

    print("\n✅ Success! Files saved.")
    print("Probability stats:")
    print("  min:", probability_map.min())
    print("  mean:", probability_map.mean())
    print("  max:", probability_map.max())
    print("  90th percentile:", np.percentile(probability_map, 90))
    print("  99th percentile:", np.percentile(probability_map, 99))

    return output_geotiff_path, output_png_path


if __name__ == '__main__':
    print("This module is intended to be imported by app.py")
