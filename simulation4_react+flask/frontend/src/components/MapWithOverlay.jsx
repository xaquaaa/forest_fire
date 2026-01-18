import { MapContainer, TileLayer } from "react-leaflet";
import CARasterOverlay from "./CARasterOverlay";
import IgnitionMarkers from "./IgnitionMarkers";
import "leaflet/dist/leaflet.css";

export default function MapWithOverlay({
  grid,
  ignitionPoints,
  onIgnitionAdd,
  fireOpacity,
  basemap
}) {
  return (
    <MapContainer
      center={[29.6, 79.6]}
      zoom={9}
      style={{ width: "100%", height: "100%" }}
    >
      {/* Base map */}
      {basemap === "osm" && (
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="© OpenStreetMap contributors"
        />
      )}

      {basemap === "satellite" && (
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution="© Esri, Maxar, Earthstar Geographics"
        />
      )}


      {/* Fire raster overlay */}
      {grid && (
        <CARasterOverlay
          grid={grid}
          ignitionPoints={ignitionPoints}
          onIgnitionAdd={onIgnitionAdd}
        />
      )}

      {/* 🔥 Ignition markers */}
      {ignitionPoints && ignitionPoints.length > 0 && (
        <IgnitionMarkers points={ignitionPoints} />
      )}
    </MapContainer>
  );
}
