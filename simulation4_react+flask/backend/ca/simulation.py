from ca.engine import ca_step

def run_simulation(
    initial_state,
    p_ignite,
    lulc,
    slope,
    aspect,
    wind_dir,
    hours=24
):
    frames = []
    state = initial_state.copy()

    for _ in range(hours):
        state = ca_step(
            state, p_ignite,
            lulc, slope, aspect, wind_dir
        )
        frames.append(state.copy())

    return frames
