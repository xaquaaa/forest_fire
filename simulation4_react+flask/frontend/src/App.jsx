import { useState, useEffect } from "react";
import { runSimulation } from "./api";
import GridCanvas from "./components/GridCanvas";
import { exportGIF } from "./api";
import { getGridMetadata } from "./api";
import "./App.css";
import MapView from "./components/MapView";
import MapWithOverlay from "./components/MapWithOverlay";
import { computeFireStats } from "./utils/fireStats";
import FireLegend from "./components/FireLegend";

const SNAPSHOT_HOURS = [1, 2, 3, 6, 12, 24];




function App() {
  
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const region = params.get("region");

    if (region) {
      console.log("Simulation region:", region);
    }
  }, []);

  const [fireOpacity, setFireOpacity] = useState(0.85);
  const [fireStats, setFireStats] = useState(null);
  const [basemap, setBasemap] = useState("osm"); // "osm" | "satellite"

  const AVAILABLE_DATES = ["2020-05-15"];

  const [frames, setFrames] = useState([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(300);
  const [userIgnition, setUserIgnition] = useState([]);
  const getToday = () => {
    return new Date().toISOString().split("T")[0];
  };

  const [selectedDate, setSelectedDate] = useState(getToday());

  const [ignitionMode, setIgnitionMode] = useState("model");
  const [gridShape, setGridShape] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);


  useEffect(() => {
    const handleFullscreenChange = () => {
      const fullscreenActive = Boolean(document.fullscreenElement);
      setIsFullscreen(fullscreenActive);

      // Force GridCanvas to recalc scale
      setCurrentFrame((f) => f);
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange);
    };
  }, []);


  const enterFullscreen = () => {
    const stage = document.querySelector(".stage-container");
    if (!stage) return;

    if (stage.requestFullscreen) {
      stage.requestFullscreen();
    } else if (stage.webkitRequestFullscreen) {
      stage.webkitRequestFullscreen(); // Safari
    } else if (stage.msRequestFullscreen) {
      stage.msRequestFullscreen(); // IE11
    }
  };

  const exitFullscreen = () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    }
  };



  const handleCellClick = (lat, lon, row, col) => {
    setUserIgnition((prev) => [
      ...prev,
      { lat, lon, row, col }
    ]);
  };

  const jumpToHour = (hour) => {
    if (frames.length === 0) return;

    const index = Math.min(hour - 1, frames.length - 1);

    setPlaying(false);        // stop autoplay
    setCurrentFrame(index);   // jump directly
  };


  const start = async () => {

    if (!AVAILABLE_DATES.includes(selectedDate)) {
      alert(
        `No dataset available for ${selectedDate}.
    Available date(s): ${AVAILABLE_DATES.join(", ")}`
      );
      return;
    }

    if (ignitionMode === "user" && userIgnition.length === 0) {
      alert("Please select at least one ignition cell.");
      return;
    }
    setPlaying(false);
    setCurrentFrame(0);

    const payload = {
      date: selectedDate,
      mode: ignitionMode
    };

    if (ignitionMode === "user") {
      payload.user_points = userIgnition;
    }

    const data = await runSimulation(payload);

    setFrames(data.frames);
    setGridShape(data.grid_shape);
  };

  const emptyGrid =
    gridShape
      ? Array.from({ length: gridShape.rows }, () =>
          Array(gridShape.cols).fill(0)
        )
      : null;
  const handleExportGIF = async () => {
    if (frames.length === 0) {
      alert("No simulation frames to export.");
      return;
    }

    try {
      await exportGIF(frames);
      alert("✅ GIF exported successfully (check backend/outputs)");
    } catch (err) {
      console.error(err);
      alert("❌ Failed to export GIF");
    }
  };
  useEffect(() => {
    const fetchGridShape = async () => {
      if (ignitionMode === "user") {
        const data = await getGridMetadata(selectedDate);
        setGridShape(data.grid_shape);
      } else {
        setGridShape(null);
      }
    };

    fetchGridShape();
  }, [ignitionMode, selectedDate]);

  useEffect(() => {
    if (!playing) return;
    if (frames.length === 0) return;

    if (currentFrame >= frames.length - 1) {
      setPlaying(false);
      return;
    }

    const timer = setTimeout(() => {
      setCurrentFrame((prev) => prev + 1);
    }, speed);

    return () => clearTimeout(timer);
  }, [playing, currentFrame, speed, frames]);

  // Update fire stats when frames or currentFrame changes
  useEffect(() => {
    if (frames.length === 0) {
      setFireStats(null);
      return;
    }

    const stats = computeFireStats(
      frames[currentFrame],
      300 // adjust if your CA resolution changes
    );

    setFireStats(stats);
  }, [frames, currentFrame]);


  return (
    <div className="app-container">
      <header className="app-header">
         Forest Fire Spread Simulation
          <div className="info-box">
            Region: Almora 
          </div>

      </header>

      <div className="app-body">
        {/* LEFT PANEL */}
        <div className="control-panel">
          <h2>Simulation Controls</h2>

          {/* Date */}
          <label className="control-label">
            Date
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
            />
          </label>

          {/* Mode */}
          <div className="control-group">
            <label>
              <input
                type="radio"
                checked={ignitionMode === "model"}
                onChange={() => setIgnitionMode("model")}
              />
              Model-based ignition
            </label>

            <label>
              <input
                type="radio"
                checked={ignitionMode === "user"}
                onChange={() => setIgnitionMode("user")}
              />
              User-defined ignition
            </label>
          </div>

          {/* Speed */}
          <label className="control-label">
            Speed (ms)
            <input
              type="range"
              min="50"
              max="1000"
              step="50"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
            />
            <span>{speed}</span>
          </label>
          {/* Fire Opacity */}
          <label className="control-label">
            Fire Overlay Opacity
            <input
              type="range"
              min="0.1"
              max="1"
              step="0.05"
              value={fireOpacity}
              onChange={(e) => setFireOpacity(Number(e.target.value))}
            />
            <span>{fireOpacity.toFixed(2)}</span>
          </label>
          {/* Basemap */}
          <div className="control-group">
            <label>
              <input
                type="radio"
                checked={basemap === "osm"}
                onChange={() => setBasemap("osm")}
              />
              OpenStreetMap
            </label>

            <label>
              <input
                type="radio"
                checked={basemap === "satellite"}
                onChange={() => setBasemap("satellite")}
              />
              Satellite
            </label>
          </div>

          {/* Buttons */}
          <div className="button-group">
            <button className="primary" onClick={start}>
              Run Simulation
            </button>

            <button
              onClick={() => setPlaying((p) => !p)}
              disabled={frames.length === 0}
            >
              {playing ? "Pause" : "Play"}
            </button>

            <button
              onClick={handleExportGIF}
              disabled={frames.length === 0}
            >
              Export GIF
            </button>

            <button
              className="danger"
              onClick={() => {
                setUserIgnition([]);
                setFrames([]);
                setCurrentFrame(0);
                setPlaying(false);
              }}
            >
              Reset
            </button>

            <button onClick={() => setUserIgnition([])}>
              Clear Ignition
            </button>
          </div>

          <div className="info-box">
            Selected ignition points: {userIgnition.length}
          </div>
          
          {fireStats && (
            <div className="info-box">
              <h4>🔥 Fire Statistics</h4>
              <p>Hour: {currentFrame + 1}</p>
              <p>Burning cells: {fireStats.burning}</p>
              <p>Burned cells: {fireStats.burned}</p>
              <p>Unburned cells: {fireStats.unburned}</p>
              <p>
                Burned Area: <b>{fireStats.burnedAreaKm2} km²</b>
              </p>
            </div>
          )}
          


        </div>

        {/* RIGHT PANEL */}
        <div className="visual-panel">
          {ignitionMode === "user" && gridShape && frames.length === 0 && (
            <>
              <h3>Select Ignition Cells</h3>
              <button
                className="secondary"
                onClick={() => {
                  isFullscreen ? exitFullscreen() : enterFullscreen();
                }}
              >
                {isFullscreen ? "⤫ Exit Fullscreen" : "⛶ Fullscreen"}
              </button>

              <div className="stage-container">
              <MapWithOverlay
                grid={frames.length > 0 ? frames[currentFrame] : emptyGrid}
                ignitionPoints={userIgnition}
                fireOpacity={fireOpacity}
                basemap={basemap}
                onIgnitionAdd={
                  ignitionMode === "user"
                    ? (lat, lon, r, c) =>
                        setUserIgnition((prev) => [
                          ...prev,
                          { row: r, col: c, lat, lon }
                        ])
                    : null
                }
              />


              </div>


            </>
          )}

          {frames.length > 0 && (
            <>
              <h3>Simulation Output</h3>
              <button
                className="secondary"
                onClick={() => {
                  isFullscreen ? exitFullscreen() : enterFullscreen();
                }}
              >
                {isFullscreen ? "⤫ Exit Fullscreen" : "⛶ Fullscreen"}
              </button>
              {frames.length > 0 && (
                <div className="time-controls">
                  {SNAPSHOT_HOURS.map((h) => (
                    <button
                      key={h}
                      className={`time-btn ${
                        currentFrame === h - 1 ? "active" : ""
                      }`}
                      onClick={() => jumpToHour(h)}
                    >
                      {h} hr
                    </button>
                  ))}
                </div>
              )}

              <div className="stage-container">
                <MapWithOverlay
                  grid={frames.length > 0 ? frames[currentFrame] : emptyGrid}
                  ignitionPoints={userIgnition}
                  fireOpacity={fireOpacity}
                  basemap={basemap}
                />
              </div>
              <FireLegend />


            </>
          )}
        </div>
      </div>
    </div>
  );


}

export default App;
