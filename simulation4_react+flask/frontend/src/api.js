// src/api.js

const API_BASE = "/api";

export async function runSimulation(payload) {
  const response = await fetch(`${API_BASE}/simulate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error("Failed to run simulation");
  }

  return await response.json();
}

export async function exportGIF(frames) {
  const res = await fetch(`${API_BASE}/export_gif`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frames }),
  });

  if (!res.ok) {
    throw new Error("Failed to export GIF");
  }

  return await res.json();
}

export async function getGridMetadata(date) {
  const res = await fetch(`${API_BASE}/grid_metadata?date=${date}`);

  if (!res.ok) {
    throw new Error("Failed to fetch grid metadata");
  }

  return await res.json();
}
