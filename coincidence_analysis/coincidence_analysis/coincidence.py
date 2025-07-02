from typing import List, Tuple
import pandas as pd

class CoincidenceAnalyzer:
    def __init__(self, detector_df: pd.DataFrame, channel_a: int = 0, channel_b: int = 1):
        """
        detector_df: combined DataFrame from DetectorData
        channel_a / channel_b: IDs of two detectors to analyze coincidences between
        """
        self.df = detector_df.sort_values("time_s").reset_index(drop=True)
        self.ch_a = channel_a
        self.ch_b = channel_b

    def find_coincidences(
        self,
        dt_window_ns: Tuple[float, float],
        energy_range_a: Tuple[int, int] = None,
        energy_range_b: Tuple[int, int] = None,
    ) -> List[Tuple[pd.Series, pd.Series]]:
        """
        Finds coincidence pairs between channel_a and channel_b where:
            dt_window_ns[0] <= (t_a - t_b) * 1e9 <= dt_window_ns[1]

        Args:
            dt_window_ns: (min_dt, max_dt) in nanoseconds.
            energy_range_a: Optional (min_energy, max_energy) for channel_a.
            energy_range_b: Optional (min_energy, max_energy) for channel_b.

        Returns:
            List of coincident (event_a, event_b) pairs.
        """
        min_dt_s = dt_window_ns[0] * 1e-9
        max_dt_s = dt_window_ns[1] * 1e-9

        df_a = self.df[self.df["channel"] == self.ch_a]
        df_b = self.df[self.df["channel"] == self.ch_b]

        if energy_range_a:
            min_ea, max_ea = energy_range_a
            df_a = df_a[(df_a["energy"] >= min_ea) & (df_a["energy"] <= max_ea)]

        if energy_range_b:
            min_eb, max_eb = energy_range_b
            df_b = df_b[(df_b["energy"] >= min_eb) & (df_b["energy"] <= max_eb)]

        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)

        coincidences = []
        i, j = 0, 0

        while i < len(df_a) and j < len(df_b):
            t_a = df_a.loc[i, "time_s"]
            t_b = df_b.loc[j, "time_s"]
            dt = t_a - t_b

            if min_dt_s <= dt <= max_dt_s:
                coincidences.append((df_a.loc[i], df_b.loc[j]))
                i += 1
                j += 1
            elif dt < min_dt_s:
                i += 1
            else:
                j += 1

        return coincidences

    def coincidence_rate(self, coincidence_window_ns: float, total_time_s: float = None) -> float:
        """
        Estimate coincidence rate (Hz) over total acquisition time.
        If total_time_s is None, infer it from the last timestamp in the dataset.
        """
        if total_time_s is None:
            total_time_s = self.df["time_s"].max() - self.df["time_s"].min()
        
        n_coincidences = len(self.find_coincidences(coincidence_window_ns))
        return n_coincidences / total_time_s if total_time_s > 0 else 0.0
