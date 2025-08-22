import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DetectorEvent:
    channel: int
    time_s: float
    energy: int
    psd: int

class DetectorData:
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths if isinstance(file_paths, list) else [file_paths]
        self.df = self._load_listmode_files(self.file_paths)

    def _load_listmode_files(self, filepaths: List[str]) -> pd.DataFrame:
        dfs = []
        for filepath in filepaths:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            header_lines = [line.strip() for line in lines if line.startswith("#")]
            data_lines = [line.strip() for line in lines if not line.startswith("#")]
            # Create DataFrame from listmode data
            df = pd.DataFrame(
                [line.split(';') for line in data_lines],
                columns=["channel", "time_s", "energy", "psd"]
            )
            df = df.astype({
                "channel": int,
                "time_s": float,
                "energy": int,
                "psd": int
            })
            df["source_file"] = filepath  # Optionally track source file
            dfs.append(df)
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
        else:
            merged_df = pd.DataFrame(columns=["channel", "time_s", "energy", "psd", "source_file"])
        return merged_df

    def get_events(self) -> List[DetectorEvent]:
        return [
            DetectorEvent(row.channel, row.time_s, row.energy, row.psd)
            for row in self.df.itertuples(index=False)
        ]

    def get_events_by_channel(self, channel_id: int) -> pd.DataFrame:
        return self.df[self.df["channel"] == channel_id].copy()

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()
    
    def get_dataframe_time_shifted(self, time_shift: float) -> pd.DataFrame:
        # Extract the last `time_shift` seconds from the dataframe
        if self.df.empty:
            return self.df.copy()
        max_time = self.df['time_s'].max()
        min_time = max_time - time_shift
        df_shifted = self.df[self.df['time_s'] >= min_time].copy()
        return df_shifted