import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.callbacks import ModelCheckpoint


import numpy as np
import os

#--------------------------------------------------

checkpoint_recall = ModelCheckpoint(
    filepath='best_recall_model.keras',
    monitor='val_recall',
    mode='max',
    save_best_only=True,
    verbose=1
)


# --- Configuration ---
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 8
DATA_DIR = "training_data"

# --- 1. Load the saved data ---
try:
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    Y_train = np.load(os.path.join(DATA_DIR, 'Y_train.npy'))
    X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
    Y_val = np.load(os.path.join(DATA_DIR, 'Y_val.npy'))

    print(f"Loaded {len(X_train)} training samples and {len(X_val)} validation samples.")
except FileNotFoundError:
    print(f"Error: Could not find training data in the '{DATA_DIR}' folder.")
    print("Please run 'preprocess.py' first to generate the data.")
    exit()


# =========================================================
# 🔥 2. WEIGHTED BINARY CROSS-ENTROPY LOSS (STABLE)
# =========================================================

def weighted_bce(y_true, y_pred):
    """
    Weighted Binary Cross-Entropy to handle class imbalance.
    Strongly penalizes missed fire pixels.
    """
    pos_weight = 6.0   # fire pixels (can tune 8–15)
    neg_weight = 1.0    # non-fire pixels

    bce = K.binary_crossentropy(y_true, y_pred)
    weights = y_true * pos_weight + (1 - y_true) * neg_weight
    return K.mean(weights * bce)





# =========================================================
# 📏 3. PROPER IoU METRIC (THRESHOLDED)
# =========================================================

def iou_thresholded(y_true, y_pred):
    """
    IoU computed after thresholding probabilities
    """
    y_pred = tf.cast(y_pred > 0.25, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return intersection / (union + 1e-7)



# =========================================================
# 🧠 4. Define the U-Net Architecture (UNCHANGED)
# =========================================================

def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoder
    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5)
    )
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6)
    )
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7)
    )
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8)
    )
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=outputs)




# =========================================================
# 🚀 5. Compile & Train
# =========================================================

model = unet_model()

model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss=weighted_bce,
    metrics=[iou_thresholded,
             Recall(thresholds = 0.25, name='recall'),
             Precision(thresholds = 0.25, name='precision')]
)


print("\n🔥 Starting model training...")

history = model.fit(
    X_train,
    Y_train,
    batch_size=2,
    epochs=25,
    validation_data=(X_val, Y_val),
    shuffle=True,
    verbose=1
)

# =========================================================
# 💾 6. Save Model
# =========================================================

model.save('unet_fire_susceptibility_model.h5')
print("\n🔥 Model training complete and saved as 'unet_fire_susceptibility_model.h5'")
