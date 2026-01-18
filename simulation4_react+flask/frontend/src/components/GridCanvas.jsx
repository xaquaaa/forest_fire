import { useRef, useEffect } from "react";

export default function GridCanvas({
  grid,
  ignitionPoints = [],
  onCellClick
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!grid || grid.length === 0) return;

    const canvas = canvasRef.current;
    if (!canvas || !canvas.parentElement) return;

    const ctx = canvas.getContext("2d");

    const rows = grid.length;
    const cols = grid[0].length;

    // 🔥 CRITICAL: canvas fills overlay div
    const width = canvas.parentElement.clientWidth;
    const height = canvas.parentElement.clientHeight;

    // Set canvas resolution to match display size
    canvas.width = width;
    canvas.height = height;

    const cellWidth = width / cols;
    const cellHeight = height / rows;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = grid[i][j];

        const isIgnition = ignitionPoints.some(
          (p) => p.row === i && p.col === j
        );


        // 🔥 Priority: ignition > fire > burned > unburned
        if (isIgnition) {
          ctx.fillStyle = "rgba(21, 101, 192, 0.9)"; // blue
        } else if (val === 1) {
          ctx.fillStyle = "rgba(211, 47, 47, 0.85)"; // burning
        } else if (val === 2) {
          ctx.fillStyle = "rgba(117, 117, 117, 0.6)"; // burned
        } else {
          // 🌱 Unburned study area (LIGHT GREEN)
          ctx.fillStyle = "rgba(76, 175, 80, 0.5)";
        }

        ctx.fillRect(
          j * cellWidth,
          i * cellHeight,
          cellWidth,
          cellHeight
        );
      }
    }

  }, [grid, ignitionPoints]);

  // ❌ Disable clicks for now (map handles interaction)
  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        height: "100%",
        background: "transparent",
        pointerEvents: "none"
      }}
    />
  );
}
