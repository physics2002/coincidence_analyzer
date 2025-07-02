import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
import pandas as pd

def plot_energy_spectrum(df: pd.DataFrame, channel: int = 0, bins: int = 200):
    """
    Plot energy spectrum for a given detector channel.
    """
    energy = df[df["channel"] == channel]["energy"]
    plt.figure(figsize=(8, 4))
    plt.hist(energy, bins=bins, alpha=0.7, label=f"Channel {channel}")
    plt.xlabel("Energy")
    plt.ylabel("Counts")
    plt.title(f"Energy Spectrum - Channel {channel}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def select_energy_range(df: pd.DataFrame, channel: int = 0, bins: int = 200) -> tuple:
    """
    Plot interactive energy spectrum where user can select a range by dragging.
    Returns:
        (min_energy, max_energy) selected by the user.
    """
    selected_range = {"min": None, "max": None}
    
    fig, ax = plt.subplots(figsize=(8, 4))
    energy = df[df["channel"] == channel]["energy"]
    ax.hist(energy, bins=bins, alpha=0.7, label=f"Channel {channel}")
    ax.set_title(f"Select energy range for channel {channel}")
    ax.set_xlabel("Energy")
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
