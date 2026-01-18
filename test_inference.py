from classify_risk import run_unet_classification
import os

input_tif = "DAILY_INPUT_DATA/almora-north_2020-05-15_input.tif"
output_tif = "output_risks/test_risk_map.tif"
output_png = "output_risks/test_risk_map.png"

os.makedirs("output_risks", exist_ok=True)

run_unet_classification(
    input_geotiff_path=input_tif,
    output_geotiff_path=output_tif,
    output_png_path=output_png
)

print("✅ Manual inference completed")
