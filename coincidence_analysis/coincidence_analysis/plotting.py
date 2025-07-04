import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pandas as pd
import numpy as np
from typing import Tuple, List

def plot_energy_spectrum(df: pd.DataFrame, channel: int = 0, bins: int = 200, energy_range: Tuple[int, int] = None):
    """
    Plot energy spectrum for a given detector channel.
    """
    energy = df[df["channel"] == channel]["energy"]
    plt.figure(figsize=(16, 8))
    plt.hist(energy, bins=bins, range=energy_range, alpha=0.7, label=f"Channel {channel}")
    plt.xlabel("Energy")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.title(f"Energy Spectrum - Channel {channel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def select_energy_range(
    df: pd.DataFrame,
    channel: int = 0,
    bins: int = 200, 
    energy_range: Tuple[int, int] = None, 
    coincidences: List[Tuple[pd.Series, pd.Series]] = None,
    preselected_range: Tuple[int, int] = None
) -> tuple:
    """
    Plot interactive energy spectrum where user can select a range by dragging.
    Returns:
        (min_energy, max_energy) selected by the user.
    """
    if preselected_range is not None:
        return preselected_range

    selected_range = {"min": None, "max": None}
    
    fig, ax = plt.subplots(figsize=(16, 8))
    energy = df[df["channel"] == channel]["energy"]
    ax.hist(energy, bins=bins, range=energy_range, alpha=0.7, label=f"Channel {channel}")

    if coincidences is not None:
        coincidence_energies = np.array([pair[channel].energy for pair in coincidences])
        ax.hist(
            coincidence_energies,
            bins=bins,
            range=energy_range,
            alpha=0.5,
            color="red",
            label="Coincidence spectrum"
        )

    ax.set_title(f"Select energy range for channel {channel}")
    ax.set_xlabel("Energy")
    ax.set_yscale("log")
    ax.set_ylabel("Counts")
    ax.legend()
    ax.grid(True)

    def onselect(xmin, xmax):
        selected_range["min"] = int(xmin)
        selected_range["max"] = int(xmax)
        print(f"Selected energy range for channel {channel}: {int(xmin)} – {int(xmax)}")

    span = SpanSelector(ax, onselect, "horizontal", useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))

    plt.tight_layout()
    plt.show()

    return selected_range["min"], selected_range["max"]

def select_time_window(delta_ts: np.ndarray, bins: int = 200, preselected_range: Tuple[float, float] = None) -> Tuple[float, float]:
    """
    Plot Δt histogram and let the user select a time window (in ns).
    Returns:
        (min_dt_ns, max_dt_ns)
    """
    if preselected_range is not None:
        return preselected_range

    selected = {"min": None, "max": None}
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.hist(delta_ts, bins=bins, alpha=0.7, color="purple")
    ax.set_title("Select coincidence time window (Δt in ns)")
    ax.set_xlabel("Δt (ns)")
    ax.set_ylabel("Counts")
    ax.grid(True)

    def onselect(xmin, xmax):
        selected["min"] = xmin
        selected["max"] = xmax
        print(f"Selected time window: {xmin:.2f} – {xmax:.2f} ns")

    span = SpanSelector(ax, onselect, "horizontal", useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='green'))

    plt.tight_layout()
    plt.show()

    return selected["min"], selected["max"]

def plot_count_rates(
    signal_ch0: dict,
    signal_ch1: dict,
    signal_coinc: dict,
    background_ch0: dict,
    background_ch1: dict,
    background_coinc: dict,
    fit_func_ch0=None, popt_ch0=None,
    fit_func_ch1=None, popt_ch1=None,
    fit_func_coinc=None, popt_coinc=None
):
    """
    Plots count rates for channel 0, channel 1, and coincidence events over time.
    """
    bin_width_s = signal_ch0['bin_width']

    def format_plots():
        plt.xlabel("Time [s]")
        plt.ylabel(f"Count Rate [1/s] (bin width = {bin_width_s:.2f}s)")
        plt.title("Count Rate vs Time")
        plt.legend()
        plt.yscale("log")
        plt.tight_layout()
        plt.grid(True)

    t_down = signal_ch0['centers'][0] - bin_width_s / 2
    t_up = signal_ch0['centers'][-1] + bin_width_s / 2

    # Plot
    plt.figure(figsize=(16, 8))
    plt.errorbar(signal_ch0['centers'], signal_ch0["rates"], yerr=signal_ch0["errors"], label="Channel 0", fmt='o', capsize=2, color='blue')
    plt.errorbar(background_ch0['centers'], background_ch0["rates"], yerr=background_ch0["errors"], label="Channel 0 Background", fmt='o', capsize=2, color='red')
    if fit_func_ch0 and popt_ch0 is not None:
        t_fit = np.linspace(t_down, t_up, 500)
        y_fit = fit_func_ch0(t_fit, *popt_ch0)
        plt.plot(t_fit, y_fit, 'g-', label='Channel 0 Fit')
    plt.xlim(t_down, t_up)
    format_plots()

    plt.figure(figsize=(16, 8))
    plt.errorbar(signal_ch1['centers'], signal_ch1["rates"], yerr=signal_ch1["errors"], label="Channel 1", fmt='o', capsize=2, color='blue')
    plt.errorbar(background_ch1['centers'], background_ch1["rates"], yerr=background_ch1["errors"], label="Channel 1 Background", fmt='o', capsize=2, color='red')
    if fit_func_ch1 and popt_ch1 is not None:
        t_fit = np.linspace(t_down, t_up, 500)
        y_fit = fit_func_ch1(t_fit, *popt_ch1)
        plt.plot(t_fit, y_fit, 'g-', label='Channel 1 Fit')
    plt.xlim(t_down, t_up)
    format_plots()

    plt.figure(figsize=(16, 8))
    plt.errorbar(signal_coinc['centers'], signal_coinc["rates"], yerr=signal_coinc["errors"], label="Coincidences", fmt='o', capsize=2, color='blue')
    plt.errorbar(background_coinc['centers'], background_coinc["rates"], yerr=background_coinc["errors"], label="Background Coincidences", fmt='o', capsize=2, color='red')
    if fit_func_coinc and popt_coinc is not None:
        t_fit = np.linspace(t_down, t_up, 500)
        y_fit = fit_func_coinc(t_fit, *popt_coinc)
        plt.plot(t_fit, y_fit, 'g-', label='Coincidence Fit')
    plt.xlim(t_down, t_up)
    format_plots()

    plt.show()