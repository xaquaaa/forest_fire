import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import os

# Paths
MODEL_PATH = 'unet_fire_susceptibility_model.h5'
DATA_DIR = 'training_data'

# Load validation data
X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
Y_val = np.load(os.path.join(DATA_DIR, 'Y_val.npy'))

# Load model
model = load_model(MODEL_PATH, compile=False)

# Predict on validation set
print("Predicting on validation data...")
Y_pred = model.predict(X_val, batch_size=4)

# Extract probabilities for fire pixels only
fire_probs = Y_pred[Y_val == 1]
nofire_probs = Y_pred[Y_val == 0]

print("\n🔥 Fire pixel probabilities:")
print(f"Mean: {fire_probs.mean():.3f}")
print(f"25% percentile: {np.percentile(fire_probs, 25):.3f}")
print(f"50% percentile (median): {np.percentile(fire_probs, 50):.3f}")
print(f"75% percentile: {np.percentile(fire_probs, 75):.3f}")

print("\n🌲 Non-fire pixel probabilities:")
print(f"Mean: {nofire_probs.mean():.3f}")
print(f"95% percentile: {np.percentile(nofire_probs, 95):.3f}")
