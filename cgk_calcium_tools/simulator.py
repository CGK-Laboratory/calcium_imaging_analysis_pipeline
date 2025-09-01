import isx
import numpy as np
from scipy.ndimage import gaussian_filter
import noise

# Gamma estimation:
# Half-decay time: 137 ± 21 ms (jGCaMP8m)
# Assuming constant 137 ms, for exponential decay:
# k = -ln(2)/137 = -0.0051 1/ms
# k = -5.1 1/s


def create_simulation(
    pixel_sizes=(600, 728),
    max_drift=7,
    n_neurons=5,
    max_cell_diameter=10,
    min_cell_diameter=7,
    seconds_time=60,
    period_ms=50,
    min_firerate=1,
    max_firerate=20,
    noise_level=0.06,
    background_scale=0.5,
    corrupted_frame_p=0.001,
    ca_level=(2, 4),
    filename="simulation.isxd",
    perlin_params=None,
):
    """
    Create a simulated calcium imaging dataset with realistic neuron activity.

    Parameters:
    -----------
    pixel_sizes : tuple, default (600, 728)
        Image dimensions as (height, width)
    max_drift : int, default 7
        Maximum drift in pixels in each direction
    n_neurons : int, default 5
        Number of neurons to simulate
    max_cell_diameter : int, default 10
        Maximum neuron diameter in pixels
    min_cell_diameter : int, default 7
        Minimum neuron diameter in pixels
    seconds_time : int, default 60
        Duration of simulation in seconds
    period_ms : int, default 50
        Frame period in milliseconds (20 Hz = 50 ms)
    min_firerate : float, default 1
        Minimum firing rate (spikes per minute)
    max_firerate : float, default 20
        Maximum firing rate (spikes per minute)
    noise_level : float, default 0.06
        Level of additive noise
    background_scale : float, default 0.5
        Scale factor for background noise
    corrupted_frame_p : float, default 0.001
        Probability of corrupted frames
    ca_level : tuple, default (2, 4)
        Calcium level range (min, max)
    filename : str, default "simulation.isxd"
        Output filename for the ISX movie
    perlin_params : dict, optional
        Parameters for Perlin noise generation. If None, uses default values.

    Returns:
    --------
    str
        Path to the created ISX movie file
    """
    # Default Perlin noise parameters
    if perlin_params is None:
        perlin_params = {
            "scale": 70.0,
            "octaves": 2,
            "persistence": 2,
            "lacunarity": 9,
            "repeatx": 100,
            "repeaty": 100,
            "base": 0,
        }

    # Convert tuple to list for easier manipulation
    pixel_sizes = list(pixel_sizes)
    ca_level = list(ca_level)

    num_samples = int(seconds_time * (1000 / period_ms))

    x = np.zeros(num_samples)
    y = np.zeros(num_samples)

    def move_and_bounce(vprev, b):
        step = np.random.choice([1, 0, -1], p=[0.01, 0.98, 0.01])
        v = vprev + step
        if v > b:
            return b - 2
        elif v < -b:
            return -b + 2
        else:
            return v

    for i in range(1, num_samples):
        x[i] = move_and_bounce(x[i - 1], max_drift)
        y[i] = move_and_bounce(y[i - 1], max_drift)

    extra_pixel_sizes = [d + max_drift * 2 for d in pixel_sizes]
    border = max_cell_diameter + max_drift
    gamma = np.exp(-5.1 * period_ms / 1000)
    cells_diameter = np.random.randint(min_cell_diameter, max_cell_diameter, n_neurons)
    cells_sigma = cells_diameter / 4

    centers = [
        [np.random.randint(border, x - border) for x in pixel_sizes]
        for i in range(n_neurons)
    ]

    trueA = np.zeros((extra_pixel_sizes + [n_neurons]), dtype=np.float32)  # area

    for i in range(n_neurons):
        tmp = np.zeros(extra_pixel_sizes)
        tmp[tuple(d // 2 for d in extra_pixel_sizes)] = 1.0
        z = np.linalg.norm(gaussian_filter(tmp, cells_sigma[i]).ravel())
        trueA[tuple(centers[i]) + (i,)] = 1
        trueA[:, :, i] = gaussian_filter(trueA[:, :, i], cells_sigma[i]) / z

    firerate = (
        np.random.rand(n_neurons, 1) * (max_firerate - min_firerate) + min_firerate
    ) / 60
    events = np.random.rand(n_neurons, num_samples) < (firerate * period_ms / 1000)
    traces = events.astype(np.float32)  # spikes
    for i in range(1, num_samples):
        traces[:, i] += gamma * traces[:, i - 1]

    cells_ca_level = (
        np.random.rand(n_neurons, 1) * (ca_level[1] - ca_level[0]) + ca_level[0]
    )
    for c in range(n_neurons):
        traces[c, :] = traces[c, :] * cells_ca_level[c] + cells_ca_level[c]

    background = np.zeros(extra_pixel_sizes)
    for i in range(extra_pixel_sizes[0]):
        for j in range(extra_pixel_sizes[1]):
            background[i][j] = noise.pnoise2(
                i / perlin_params["scale"],
                j / perlin_params["scale"],
                octaves=perlin_params["octaves"],
                persistence=perlin_params["persistence"],
                lacunarity=perlin_params["lacunarity"],
                repeatx=perlin_params["repeatx"],
                repeaty=perlin_params["repeaty"],
                base=perlin_params["base"],
            )

    # scale noise
    background = background - np.min(background)
    maxb = np.max(background)
    background = background / maxb * background_scale

    # background + traces
    data = background[:, :, None] + trueA.dot(traces)

    # drift
    data_drift = np.zeros((pixel_sizes + [num_samples]), dtype=np.float32)

    for i in range(num_samples):
        xd = int(max_drift + x[i])
        yd = int(max_drift + y[i])
        if np.random.random() < corrupted_frame_p:
            data_drift[:, :, i] = data.max()
        else:
            data_drift[:, :, i] = data[
                xd : xd + pixel_sizes[0], yd : yd + pixel_sizes[1], i
            ]

    # sum iid noise
    data_drift = data_drift + noise_level * np.random.rand(
        np.prod(pixel_sizes + [num_samples])
    ).reshape(pixel_sizes + [num_samples])

    # creates isx
    timing = isx.Timing(
        num_samples=num_samples, period=isx.Duration.from_msecs(period_ms)
    )
    spacing = isx.Spacing(num_pixels=pixel_sizes)
    movie = isx.Movie.write(filename, timing, spacing, np.float32)
    for i in range(timing.num_samples):
        movie.set_frame_data(i, data_drift[:, :, i].astype(np.float32))
    movie.flush()
    del movie

    return filename


if __name__ == "__main__":
    # Example usage
    output_file = create_simulation()
    print(f"Simulation created: {output_file}")
