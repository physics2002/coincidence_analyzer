import numpy as np

def get_times_in_energy_range(df, channel, energy_range):
    """
    Return time values for a given channel within specified energy range.
    """
    mask = (
        (df["channel"] == channel) &
        (df["energy"] >= energy_range[0]) &
        (df["energy"] <= energy_range[1])
    )
    return df[mask]["time_s"].values

def compute_count_rate(
    times: list[float],
    bin_width_s: float,
    t_min: float = None,
    t_max: float = None
) -> dict:
    """
    Compute count rate histogram and Poisson errors.
    Returns a dictionary with rates, errors, bin centers, counts.
    """
    if len(times) == 0:
        return {
            "centers": np.array([]),
            "rates": np.array([]),
            "errors": np.array([]),
            "counts": np.array([]),
            "bin_width": bin_width_s,
            "bins": np.array([])
        }

    if t_min is None:
        t_min = min(times)
    if t_max is None:
        t_max = max(times)

    bins = np.arange(t_min, t_max + bin_width_s, bin_width_s)
    centers = (bins[:-1] + bins[1:]) / 2
    counts, _ = np.histogram(times, bins=bins)

    rates = counts / bin_width_s
    errors = np.sqrt(counts) / bin_width_s

    return {
        "centers": centers,
        "rates": rates,
        "errors": errors,
        "counts": counts,
        "bin_width": bin_width_s,
        "bins": bins
    }