import rasterio

# Path to ANY Almora raster (use fire raster or LULC raster)
raster_path = "C:/Users/ALIENWARE/Documents/amil/education/GEC KKD COLLEGE/project/forest_GEE/simulation4_react+flask/backend/data/static/Almora_LULC_ESA_WorldCover_30m.tif"

with rasterio.open(raster_path) as src:
    bounds = src.bounds
    crs = src.crs

print("CRS:", crs)
print("Bounds:")
print("West (min lon):", bounds.left)
print("South (min lat):", bounds.bottom)
print("East (max lon):", bounds.right)
print("North (max lat):", bounds.top)
