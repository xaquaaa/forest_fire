import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lstm.model import FireLSTM
import os

SEQUENCE_LENGTH = 24
FEATURES = ["temp_C", "RH", "wind_speed", "rain"]

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "fire_lstm_weighted.pth")

# Load model ONCE
model = FireLSTM()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def predict_p_ignite(weather_csv, target_date):
    """
    weather_csv: path to ERA5 CSV
    target_date: 'YYYY-MM-DD'
    """

    df = pd.read_csv(weather_csv)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # Filter past 24 hours before date
    cutoff = pd.to_datetime(target_date)
    history = df[df["time"] < cutoff].tail(SEQUENCE_LENGTH)

    if len(history) < SEQUENCE_LENGTH:
        raise ValueError("Not enough weather history for LSTM")

    scaler = StandardScaler()
    X = scaler.fit_transform(history[FEATURES])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logit = model(X)
        prob = torch.sigmoid(logit).item()
        print("Raw LSTM probability:", prob)

    # 🔥 CA-safe calibration
    p_ignite = max(0.02, min(prob * 0.4, 0.18))

    print("Calibrated p_ignite:", p_ignite)
    return p_ignite


if __name__ == "__main__":
    # 🔍 DIAGNOSTIC TEST ONLY

    WEATHER_CSV = os.path.join(
        BASE_DIR,
        "..",
        "data",
        "Almora_ERA5_Hourly_Weather_2018_2020_MAM.csv"
    )

    test_date = "2020-05-15"

    print("=== LSTM DIAGNOSTIC RUN ===")
    print("CSV exists?", os.path.exists(WEATHER_CSV))

    p = predict_p_ignite(WEATHER_CSV, test_date)

    print("Returned p_ignite:", p)
