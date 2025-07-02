import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coincidence_analysis.detector import DetectorData
from coincidence_analysis.coincidence import CoincidenceAnalyzer
from coincidence_analysis.plotting import (
    plot_energy_spectrum, select_energy_range, select_time_window
)

# Load and filter
det = DetectorData("Z:\Studenten\Bakhodirov\coincidence_analyzer-1\coincidence_analysis\Detector_data\AlSurf280um_listmode.txt")
df = det.get_dataframe()
analyzer = CoincidenceAnalyzer(df)

# Energy selection
range0 = select_energy_range(df, channel=0, bins=1000, energy_range=(0, 50000))
range1 = select_energy_range(df, channel=1, bins=2000, energy_range=(0, 50000))

# Δt histogram and window selection
delta_ts = analyzer.delta_t_distribution(
    max_time_diff_ns=200
)
dt_window = select_time_window(delta_ts)

# Coincidence analysis
coincidences = analyzer.find_coincidences(
    dt_window_ns=dt_window,
    energy_range_a=range0,
    energy_range_b=range1
)

print(f"Found {len(coincidences)} coincidences in dt window {dt_window}.")
