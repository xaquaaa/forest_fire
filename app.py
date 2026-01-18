from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS 
from classify_risk import run_unet_classification 
import os
from datetime import datetime

# NOTE: You MUST install Pillow (PIL) for the PNG generation
# pip install Flask-Cors Pillow
print("RUNNING PHASE 1 BACKEND")

app = Flask(__name__, static_folder='output_risks', static_url_path='/static/riskmaps')
CORS(app) 

# --- SERVER CONFIGURATION ---
DATASET_DIR = "DAILY_INPUT_DATA/" 
OUTPUT_DIR = "output_risks/" 

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/api/wildfire/predict', methods=['POST'])
def get_risk_map():
    """
    Triggers prediction, generates GeoTIFF (for download) and PNG (for display), 
    and returns a JSON response with the PNG image URL and GeoTIFF filename.
    """
    try:
        data = request.get_json()
        zone_id = data.get('zoneId') 
        date_str = data.get('date')  

        if not zone_id or not date_str:
            return jsonify({"error": "Missing zone ID or date for prediction."}), 400

        # Construct file paths
        input_file_name = f"{zone_id}_{date_str}_input.tif"
        input_file_path = os.path.join(DATASET_DIR, input_file_name)
        
        output_geotiff_name = f"risk_map_{zone_id}_{date_str}.tif"
        output_geotiff_path = os.path.join(OUTPUT_DIR, output_geotiff_name)
        
        output_png_name = f"risk_map_{zone_id}_{date_str}.png" # NEW PNG FILE NAME
        output_png_path = os.path.join(OUTPUT_DIR, output_png_name)


        # 1. Check for the pre-made dataset file
        if not os.path.exists(input_file_path):
             return jsonify({
                 "error": f"Dataset for zone '{zone_id}' on {date_str} not found."
             }), 404
        
        # 2. Execute the U-Net Classification (Generates both .tif and .png)
        print(f"Executing U-Net for {zone_id} on {date_str}...")
        
        run_unet_classification(
            input_geotiff_path=input_file_path, 
            output_geotiff_path=output_geotiff_path,
            output_png_path=output_png_path # Pass PNG path to the function
        )
        
        # 3. Return JSON response with the PNG URL and GeoTIFF filename
        # The PNG URL points to the static file server route we defined in Flask init
        png_url = url_for('static', filename=output_png_name, _external=True)
        # Fix URL to include port if not running on default ports
        png_url = png_url.replace('http://127.0.0.1/', 'http://localhost:5000/')

        return jsonify({
            "message": "Prediction successful.",
            "png_url": png_url,
            "geotiff_filename": output_geotiff_name
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except IOError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")
        return jsonify({"error": f"Internal server error: {e}"}), 500

@app.route('/api/wildfire/download/<filename>', methods=['GET'])
def download_geotiff(filename):
    """
    Dedicated endpoint to handle GeoTIFF download based on the filename provided 
    by the /predict response.
    """
    if not filename.endswith('.tif'):
        return jsonify({"error": "Invalid file type requested for download."}), 400
        
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": f"GeoTIFF file not found: {filename}"}), 404
        
    return send_file(
        file_path, 
        as_attachment=True, 
        download_name=filename, 
        mimetype='image/tiff'
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )
