from typing import Tuple, Optional
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

    def _get_filtered(self, channel: int, energy_range: Tuple[int, int] = None) -> pd.DataFrame:
        df = self.df[self.df["channel"] == channel]
        if energy_range:
            min_e, max_e = energy_range
            df = df[(df["energy"] >= min_e) & (df["energy"] <= max_e)]
        return df.reset_index(drop=True)

    def coincidence_search(
        self,
        dt_window_ns: Tuple[float, float] = (-200, 200),
        energy_range_a: Tuple[int, int] = None,
        energy_range_b: Tuple[int, int] = None,
        return_delta_t: bool = False
    ):
        """
        Finds coincidence pairs or delta_t values between channel_a and channel_b.
        If return_delta_t is True, returns array of delta_t (ns) for all pairs within window.
        Otherwise, returns list of (event_a, event_b) pairs.
        """
        from coincidence_analysis.detector import DetectorEvent
        columns = list(DetectorEvent.__annotations__.keys())

        min_dt_s = dt_window_ns[0] * 1e-9
        max_dt_s = dt_window_ns[1] * 1e-9

        df_a = self._get_filtered(self.ch_a, energy_range_a)
        df_b = self._get_filtered(self.ch_b, energy_range_b)

        df_a = df_a.rename(columns={col: f"{col}_a" for col in columns})
        df_b = df_b.rename(columns={col: f"{col}_b" for col in columns})

        # Merge asof finds nearest event in df_b for each event in df_a
        merged = pd.merge_asof(
            df_a, df_b, 
            left_on="time_s_a", right_on="time_s_b",
            direction="nearest",
            tolerance=max(abs(min_dt_s), abs(max_dt_s))
        )

        # Calculate time difference
        merged = merged.dropna(subset=["channel_b"])
        dt = merged["time_s_a"] - merged["time_s_b"]

        # Select only those within the window
        mask = (dt >= min_dt_s) & (dt <= max_dt_s)
        merged = merged[mask]

        if return_delta_t:
            return (dt[mask] * 1e9).values
        else:
            # Return list of (row_a, row_b) as before
            df_a_clean = merged[[f"{col}_a" for col in columns]].copy()
            df_b_clean = merged[[f"{col}_b" for col in columns]].copy()
            df_a_clean.columns = columns
            df_b_clean.columns = columns

            return list(zip(
                df_a_clean.itertuples(index=False),
                df_b_clean.itertuples(index=False)
            ))

    def coincidence_rate(self, coincidence_window_ns: float, total_time_s: float = None) -> float:
        """
        Estimate coincidence rate (Hz) over total acquisition time.
        If total_time_s is None, infer it from the last timestamp in the dataset.
        """
        if total_time_s is None:
            total_time_s = self.df["time_s"].max() - self.df["time_s"].min()
        
        n_coincidences = len(self.find_coincidences(coincidence_window_ns))
        return n_coincidences / total_time_s if total_time_s > 0 else 0.0