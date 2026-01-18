import { useEffect, useRef } from "react";
import L from "leaflet";
import { useMap } from "react-leaflet";
import { ALMORA_BOUNDS } from "../config/geoBounds";

export default function CARasterOverlay({ grid, ignitionPoints, onIgnitionAdd, fireOpacity = 0.85 }) {
  const map = useMap();
  const overlayRef = useRef(null);
  const canvasRef = useRef(document.createElement("canvas"));

  useEffect(() => {
    if (!grid) return;

    const rows = grid.length;
    const cols = grid[0].length;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = cols;
    canvas.height = rows;

    ctx.clearRect(0, 0, cols, rows);

    for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
        const val = grid[i][j];

        const isIgnition = ignitionPoints.some(
        (p) => p.row === i && p.col === j
        );


        // 🔥 Priority order
        if (isIgnition) {
        ctx.fillStyle = "rgba(21, 101, 192, 1)"; // blue
        } 
        else if (val === 1) {
        ctx.fillStyle = "rgba(211, 47, 47, 0.9)"; // burning
        } 
        else if (val === 2) {
        ctx.fillStyle = "rgba(117, 117, 117, 0.7)"; // burned
        } 
        else {
        // 🌱 UNBURNED STUDY AREA (THIS WAS MISSING)
        ctx.fillStyle = "rgba(76, 175, 80, 0.5)";
        }

        ctx.fillRect(j, i, 1, 1);
    }
    }


    const imageUrl = canvas.toDataURL();

    if (overlayRef.current) {
      map.removeLayer(overlayRef.current);
    }

    overlayRef.current = L.imageOverlay(
      imageUrl,
      [
        [ALMORA_BOUNDS.south, ALMORA_BOUNDS.west],
        [ALMORA_BOUNDS.north, ALMORA_BOUNDS.east]
      ],
      { opacity: fireOpacity }
    ).addTo(map);
    

    return () => {
      if (overlayRef.current) {
        map.removeLayer(overlayRef.current);
      }
    };
  }, [grid, ignitionPoints, fireOpacity,map]);

    useEffect(() => {
    if (!onIgnitionAdd || !grid) return;

    const handleClick = (e) => {
        const { lat, lng } = e.latlng;

        const rows = grid.length;
        const cols = grid[0].length;

        // Bounds check
        if (
        lng < ALMORA_BOUNDS.west ||
        lng > ALMORA_BOUNDS.east ||
        lat < ALMORA_BOUNDS.south ||
        lat > ALMORA_BOUNDS.north
        ) {
        return;
        }

        const col = Math.floor(
        ((lng - ALMORA_BOUNDS.west) /
            (ALMORA_BOUNDS.east - ALMORA_BOUNDS.west)) *
            cols
        );

        const row = Math.floor(
        ((ALMORA_BOUNDS.north - lat) /
            (ALMORA_BOUNDS.north - ALMORA_BOUNDS.south)) *
            rows
        );

        // 🔥 FIXED LINE
        onIgnitionAdd(lat, lng, row, col);
    };

    map.on("click", handleClick);
    return () => {
        map.off("click", handleClick);
    };
    }, [map, grid, onIgnitionAdd]);



  return null;
}
