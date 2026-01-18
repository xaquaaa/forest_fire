import numpy as np
from ca.terrain import (
    fuel_factor, slope_factor, aspect_factor,
    neighbor_direction, wind_factor
)

def get_neighbors(i, j, shape):
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                yield ni, nj

def ca_step(state, p_ignite, lulc, slope, aspect, wind_dir):
    new_state = state.copy()
    rows, cols = state.shape

    for i in range(rows):
        for j in range(cols):

            if state[i, j] == 0:
                for ni, nj in get_neighbors(i, j, state.shape):
                    if state[ni, nj] == 1:

                        P = (
                            p_ignite
                            * fuel_factor(lulc[i, j])
                            * slope_factor(slope[i, j])
                            * aspect_factor(aspect[i, j])
                            * wind_factor(
                                neighbor_direction(i, j, ni, nj),
                                wind_dir
                            )
                        )
                        P = min(P, 1.0)

                        if np.random.rand() < P:
                            new_state[i, j] = 1
                            break

            elif state[i, j] == 1:
                new_state[i, j] = 2

    return new_state
