import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
MODEL_PATH = 'unet_fire_susceptibility_model.h5'
DATA_DIR = 'training_data'
VISUAL_OUTPUT_FILE = 'model_validation_comparison.png'
GRAPH_OUTPUT_FILE = 'validation_metrics_barchart.png'

LULC_BAND_INDEX = 3

PREDICTOR_BAND_NAMES = [
    '1. Elevation', '2. Slope_Deg', '3. Aspect_Deg', '4. LULC_Class',
    '5. GHSL_Pop', '6. Avg_MAM_Temp_C', '7. Total_MAM_Precip_mm', '8. Avg_MAM_Wind_Deg'
]

# --------------------------------------------------
# CUSTOM OBJECTS (MUST MATCH TRAINING)
# --------------------------------------------------
def iou_thresholded(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.3, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def weighted_bce(y_true, y_pred):
    pos_weight = 10.0
    neg_weight = 1.0
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weights = y_true * pos_weight + (1 - y_true) * neg_weight
    return tf.reduce_mean(weights * bce)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading trained U-Net model...")

model = load_model(
    MODEL_PATH,
    custom_objects={
        'weighted_bce': weighted_bce,
        'iou_thresholded': iou_thresholded
    },
    compile=False
)

print("✅ Model loaded successfully")

# --------------------------------------------------
# LOAD VALIDATION DATA
# --------------------------------------------------
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
Y_val = np.load(os.path.join(DATA_DIR, 'Y_val.npy'))

print(f"Validation samples: {len(X_val)}")

# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------
print("\nRunning model predictions...")
Y_pred_prob = model.predict(X_val, batch_size=2)
Y_pred_binary = (Y_pred_prob > 0.25).astype(np.float32)

# --------------------------------------------------
# MANUAL METRIC CALCULATION (ROBUST)
# --------------------------------------------------
y_true = Y_val.reshape(-1)
y_pred = Y_pred_binary.reshape(-1)

TP = np.sum((y_true == 1) & (y_pred == 1))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

# Accuracy (overall pixel accuracy)
accuracy = np.mean(y_true == y_pred)

precision = TP / (TP + FP + 1e-7)
recall = TP / (TP + FN + 1e-7)
f1_score = 2 * precision * recall / (precision + recall + 1e-7)

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1_score]
}).round(4)


print("\n#################################################")
print("# Validation Metrics (Imbalance-Aware)")
print("#################################################")
print(metrics_df.to_markdown(index=False))
print("#################################################")

# --------------------------------------------------
# BAR CHART
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.bar(metrics_df['Metric'], metrics_df['Value'],
        color=['gold', 'lightgreen', 'skyblue'])
plt.ylim(0, 1)
plt.title('Validation Metrics (Fire Class)', fontsize=14)

for i, v in enumerate(metrics_df['Value']):
    plt.text(i, v + 0.02, f"{v:.4f}", ha='center')

plt.savefig(GRAPH_OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Metrics bar chart saved: {GRAPH_OUTPUT_FILE}")

# --------------------------------------------------
# QUALITATIVE VISUALIZATION
# --------------------------------------------------
print("\nGenerating qualitative validation samples...")

num_samples = 5
indices = random.sample(range(len(X_val)), num_samples)

fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

for i, idx in enumerate(indices):
    axes[i, 0].imshow(Y_val[idx].squeeze(), cmap='gray')
    axes[i, 0].set_title("Ground Truth")
    axes[i, 0].axis('off')

    axes[i, 1].imshow(X_val[idx, :, :, LULC_BAND_INDEX], cmap='tab10')
    axes[i, 1].set_title("Input: LULC")
    axes[i, 1].axis('off')

    axes[i, 2].imshow(Y_pred_binary[idx].squeeze(), cmap='Reds')
    axes[i, 2].set_title("Prediction")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig(VISUAL_OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Visual comparison saved: {VISUAL_OUTPUT_FILE}")
