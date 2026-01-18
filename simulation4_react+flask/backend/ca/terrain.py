import numpy as np

FUEL_MAP = {
    10: 1.4, 20: 1.2, 30: 1.1, 40: 0.7,
    50: 0.1, 60: 0.2, 80: 0.0,
    90: 0.4, 95: 0.0, 100: 0.1
}

def fuel_factor(lulc_value):
    return FUEL_MAP.get(int(lulc_value), 0.5)

def slope_factor(slope_deg):
    if not np.isfinite(slope_deg):
        return 1.0
    return 1.0 + (slope_deg / 30.0)

def aspect_factor(aspect_deg):
    if np.isnan(aspect_deg):
        return 1.0
    return 1.2 if 135 <= aspect_deg <= 225 else 1.0

def wind_direction(u, v):
    return (np.degrees(np.arctan2(u, v)) + 360) % 360

def neighbor_direction(i, j, ni, nj):
    return (np.degrees(np.arctan2(j - nj, ni - i)) + 360) % 360

def wind_factor(cell_dir, wind_dir):
    diff = abs(cell_dir - wind_dir)
    diff = min(diff, 360 - diff)
    return 1.0 + 0.6 * np.cos(np.radians(diff))
