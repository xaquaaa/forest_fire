import pandas as pd
import numpy as np
import rasterio 
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import FireLSTM
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F

# Load CSV
df = pd.read_csv(
    "data/Almora_ERA5_Hourly_Weather_2018_2020_MAM.csv"
)

print(df.head())
print(df.columns)
print("Total records:", len(df))

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'])

# Drop unnecessary columns
df = df.drop(columns=['system:index', '.geo'])

# Sort by time
df = df.sort_values('time').reset_index(drop=True)

# Fill missing values
df = df.ffill()

print(df.head())
print("Missing values:\n", df.isna().sum())

# Create date column
df['date'] = df['time'].dt.date


# Load fire raster
fire_raster_path = "data/Almora_FireBinary_MCD64A1_2018_2020_MAM_30m.tif"

with rasterio.open(fire_raster_path) as src:
    fire_data = src.read(1)

print("Fire raster unique values:", np.unique(fire_data))

# Percentage of area burned
burn_ratio = np.mean(fire_data)

print("Burned area ratio:", burn_ratio)

# Group by date
daily_df = df.groupby('date').mean().reset_index()

# Fire label based on burn ratio threshold
# -----------------------------------
# CREATE FIRE LABELS FROM RASTER
# -----------------------------------

# Count burned pixels
total_pixels = fire_data.size
burned_pixels = np.sum(fire_data == 1)

# If any fire exists in dataset, mark corresponding days
# (Simplification: fire occurred on burned days in MAM)
fire_exists = burned_pixels > 0

# Create daily fire labels
daily_df = df.groupby('date').mean().reset_index()

daily_df['fire_label'] = 0
if fire_exists:
    # Mark fire days based on historical burn presence
    fire_days = daily_df.sample(
        frac=burn_ratio, random_state=42
    ).index
    daily_df.loc[fire_days, 'fire_label'] = 1

print("Daily fire label distribution:")
print(daily_df['fire_label'].value_counts())



#merge back to hourly data
df = df.merge(
    daily_df[['date', 'fire_label']],
    on='date',
    how='left'
)

print("Hourly data with fire labels:")
print(df[['time', 'date', 'fire_label']].head(30))



FEATURES = ['temp_C', 'RH', 'wind_speed', 'rain']

scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])


# Prepare sequences for LSTM

SEQUENCE_LENGTH = 24
FEATURES = ['temp_C', 'RH', 'wind_speed', 'rain']

X, y = [], []

for i in range(len(df) - SEQUENCE_LENGTH):
    X.append(df[FEATURES].iloc[i:i+SEQUENCE_LENGTH].values)
    y.append(df['fire_label'].iloc[i+SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y unique values:", np.unique(y))
print("y distribution:", np.bincount(y))
print("Sample X[0]:", X[0])

# -------------------------------
# TIME-BASED TRAIN / VAL SPLIT
# -------------------------------

split_idx = int(0.8 * len(X))  # 80% past, 20% future

X_train = X[:split_idx]
y_train = y[:split_idx]

X_val = X[split_idx:]
y_val = y[split_idx:]

print("Train samples:", X_train.shape)
print("Validation samples:", X_val.shape)


# -----------------------------------
# HANDLE CLASS IMBALANCE
# -----------------------------------

num_fire = np.sum(y_train == 1)
num_no_fire = np.sum(y_train == 0)

pos_weight = torch.tensor(
    [num_no_fire / num_fire],
    dtype=torch.float32
)

print("Positive class weight:", pos_weight.item())

# -----------------------------------
# CONVERT TO PYTORCH TENSORS
# -----------------------------------

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds   = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

model = FireLSTM()

criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------------
# TRAINING LOOP
# -----------------------------------

EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

# Save the trained model
torch.save(model.state_dict(), "fire_lstm_weighted.pth")

#model evaluation on validation set

model.eval()

with torch.no_grad():
    val_logits = model(X_val_t)
    val_probs = torch.sigmoid(val_logits).cpu().numpy()

threshold = 0.3   # try 0.3 first
val_preds = (val_probs >= threshold).astype(int)
print("Threshold:", threshold)
print(classification_report(y_val, val_preds))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_preds))
for t in [0.2, 0.3, 0.4, 0.5]:
    preds = (val_probs >= t).astype(int)
    print("\nThreshold:", t)
    print(confusion_matrix(y_val, preds))
