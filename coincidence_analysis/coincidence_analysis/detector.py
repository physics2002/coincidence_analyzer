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
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df, self.metadata = self._load_listmode_file(file_path)

    def _load_listmode_file(self, filepath: str) -> Tuple[pd.DataFrame, dict]:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header_lines = [line.strip() for line in lines if line.startswith("#")]
        data_lines = [line.strip() for line in lines if not line.startswith("#")]

        metadata = {}
        for line in header_lines:
            if "Sample interval" in line:
                metadata["sample_interval_ns"] = float(line.split(":")[1].split()[0])
            elif "Gate length" in line:
                parts = line.split(":")[1].strip().split()
                metadata["gate_ch0"] = int(parts[1])
                metadata["gate_ch1"] = int(parts[3])

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

        return df, metadata

    def get_metadata(self) -> dict:
        return self.metadata

    def get_events(self) -> List[DetectorEvent]:
        return [
            DetectorEvent(row.channel, row.time_s, row.energy, row.psd)
            for row in self.df.itertuples(index=False)
        ]

    def get_events_by_channel(self, channel_id: int) -> pd.DataFrame:
        return self.df[self.df["channel"] == channel_id].copy()

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.copy()