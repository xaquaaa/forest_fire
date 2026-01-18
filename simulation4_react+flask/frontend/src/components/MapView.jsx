import { MapContainer, TileLayer, useMap } from "react-leaflet";
import { useEffect } from "react";
import "leaflet/dist/leaflet.css";
import "leaflet-defaulticon-compatibility";

function ResizeFix({ isFullscreen }) {
  const map = useMap();

  useEffect(() => {
    const timeout = setTimeout(() => {
      map.invalidateSize(true);
    }, 300);

    return () => clearTimeout(timeout);
  }, [map, isFullscreen]);

  return null;
}

export default function MapView({ isFullscreen }) {
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <MapContainer
        center={[29.6, 79.7]}
        zoom={9}
        style={{ width: "100%", height: "100%" }}
      >
        <ResizeFix isFullscreen={isFullscreen} />
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="© OpenStreetMap contributors"
        />
      </MapContainer>
    </div>
  );
}
