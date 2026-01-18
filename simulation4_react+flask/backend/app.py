from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image 
import numpy as np
import rasterio
import os
from flask import Flask, send_from_directory, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
print("RUNNING PHASE 2 BACKEND")


from ca.simulation import run_simulation
from ca.terrain import wind_direction
from ca.data_loader import load_data_for_date

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path=""
)
print("STATIC FOLDER:", app.static_folder)

CORS(app)  # allow React to call Flask


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400
    date = data["date"]          # "2020-05-15"
    ignition_mode = data["mode"] # "model" | "user"
    user_points = data.get("user_points", [])

    if not date or not ignition_mode:
        return jsonify({"error": "Missing date or mode"}), 400

    date = date.replace("-", "_")

    p_ignite, fire_seed = load_data_for_date(date)

    DOWNSAMPLE = 10
    BASE_DIR = os.path.dirname(__file__)

    def load_static(name):
        path = os.path.join(BASE_DIR, "data", "static", name)
        with rasterio.open(path) as src:
            return src.read(1)[::DOWNSAMPLE, ::DOWNSAMPLE]

    lulc = load_static("Almora_LULC_ESA_WorldCover_30m.tif")
    slope = np.nan_to_num(load_static("Almora_Slope_SRTM_30m.tif"))
    aspect = load_static("Almora_Aspect_SRTM_30m.tif")

    state = np.zeros_like(fire_seed[::DOWNSAMPLE, ::DOWNSAMPLE], dtype=np.uint8)
    rows, cols = state.shape

    if ignition_mode == "model":
        state[fire_seed[::DOWNSAMPLE, ::DOWNSAMPLE] == 1] = 1
    else:
        for p in user_points:
            i = p["row"]
            j = p["col"]
            state[i, j] = 1


    wind_dir = wind_direction(2.0, 1.0)

    frames = run_simulation(
        state, p_ignite,
        lulc, slope, aspect, wind_dir,
        hours=24
    )

    # Convert frames to lists (JSON-safe)
    frames_json = [f.tolist() for f in frames]

    return jsonify({
        "frames": frames_json,
        "grid_shape": {
            "rows": rows,
            "cols": cols
        }
    })


# Export GIF endpoint
@app.route("/api/export_gif", methods=["POST"])
def export_gif():
    frames = request.json["frames"]

    images = []
    for frame in frames:
        img = Image.fromarray(
            (np.array(frame) * 120).astype(np.uint8)
        )
        images.append(img)

    path = "outputs/fire_simulation.gif"
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=300,
        loop=0
    )

    return jsonify({"path": path})

# Grid metadata endpoint
@app.route("/api/grid_metadata", methods=["GET"])
def grid_metadata():
    date = request.args.get("date")

    if not date:
        return jsonify({"error": "date is required"}), 400

    # 🔥 FIX: normalize date to match folder
    date = date.replace("-", "_")

    _, fire_seed = load_data_for_date(date)

    DOWNSAMPLE = 10
    state = fire_seed[::DOWNSAMPLE, ::DOWNSAMPLE]
    rows, cols = state.shape

    return jsonify({
        "grid_shape": {
            "rows": rows,
            "cols": cols
        }
    })



@app.route("/")
def serve_react():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_spa(path):
    full_path = os.path.join(app.static_folder, path)
    if os.path.exists(full_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")





if __name__ == "__main__":
    
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True,
        use_reloader=False
    )

    
