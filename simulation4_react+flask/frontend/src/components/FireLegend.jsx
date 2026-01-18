import "./FireLegend.css";

export default function FireLegend() {
  return (
    <div className="fire-legend">
      <h4>Fire Legend</h4>

      <div className="legend-item">
        <span className="legend-color unburned"></span>
        <span>Unburned</span>
      </div>

      <div className="legend-item">
        <span className="legend-color burning"></span>
        <span>Burning</span>
      </div>

      <div className="legend-item">
        <span className="legend-color burned"></span>
        <span>Burned</span>
      </div>

      <div className="legend-item">
        <span className="legend-color ignition"></span>
        <span>Ignition Point</span>
      </div>
    </div>
  );
}
