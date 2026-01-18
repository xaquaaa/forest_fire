import { Marker, Popup } from "react-leaflet";
import L from "leaflet";

const ignitionIcon = new L.Icon({
  iconUrl: "/pin.png",   // 👈 LOCAL, SAFE
  iconSize: [28, 28],
  iconAnchor: [14, 28]
});


export default function IgnitionMarkers({ points }) {
  return (
    <>
      {points.map((p, idx) => (
        <Marker
          key={idx}
          position={[p.lat, p.lon]}
          icon={ignitionIcon}
        >
          <Popup>
            🔥 Ignition Point<br />
            Row: {p.row}, Col: {p.col}
          </Popup>
        </Marker>
      ))}
    </>
  );
}
