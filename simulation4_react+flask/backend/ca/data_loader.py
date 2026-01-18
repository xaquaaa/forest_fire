import numpy as np
import rasterio
import os

# ca/data_loader.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 ca/ → backend/
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# 🔥 backend/data
DATA_DIR = os.path.join(BACKEND_DIR, "data")

print("DATA_DIR resolved to:", DATA_DIR)


def load_data_for_date(date_str):
    """
    Loads all required inputs for a given simulation date.
    Accepts flexible date formats.
    """

    # Generate possible folder name candidates
    candidates = []

    # Original
    candidates.append(date_str)

    # YYYY-MM-DD <-> YYYY_MM_DD
    candidates.append(date_str.replace("-", "_"))
    candidates.append(date_str.replace("_", "-"))

    # Handle DD-MM-YYYY → YYYY_MM_DD
    parts = date_str.replace("_", "-").split("-")
    if len(parts) == 3 and len(parts[0]) == 2:
        d, m, y = parts
        candidates.append(f"{y}_{m}_{d}")
        candidates.append(f"{y}-{m}-{d}")

    # Remove duplicates
    candidates = list(dict.fromkeys(candidates))

    # Try each candidate
    base_path = None
    for c in candidates:
        path = os.path.join(DATA_DIR, c)
        print("Trying:", path)
        if os.path.exists(path):
            base_path = path
            break

    if base_path is None:
        raise FileNotFoundError(
            f"No data found for date: {date_str}. Tried: {candidates}"
        )

    # ---- LOAD DATA ----

    ignition_path = os.path.join(base_path, "ignition_prob.npy")
    fire_path = os.path.join(base_path, "fire_seed.tif")

    if not os.path.exists(ignition_path):
        raise FileNotFoundError(f"Missing ignition_prob.npy in {base_path}")

    if not os.path.exists(fire_path):
        raise FileNotFoundError(f"Missing fire_seed.tif in {base_path}")

    p_ignite = np.load(ignition_path).item()

    with rasterio.open(fire_path) as src:
        fire_seed = src.read(1)

    return p_ignite, fire_seed
