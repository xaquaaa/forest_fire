export function computeFireStats(grid, cellSizeMeters = 300) {
  if (!grid || grid.length === 0) return null;

  let burning = 0;
  let burned = 0;
  let unburned = 0;

  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[0].length; j++) {
      if (grid[i][j] === 0) unburned++;
      else if (grid[i][j] === 1) burning++;
      else if (grid[i][j] === 2) burned++;
    }
  }

  const cellAreaKm2 =
    (cellSizeMeters * cellSizeMeters) / 1_000_000;

  return {
    burning,
    burned,
    unburned,
    burnedAreaKm2: (burned * cellAreaKm2).toFixed(2),
    totalCells: grid.length * grid[0].length
  };
}
