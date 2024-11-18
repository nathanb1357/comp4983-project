from re import split
from typing import Dict
import pandas as pd
import numpy as np
from enums import GraphType


class Split:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.df = self._load_data()
        self.columns = self._csv_to_dict()

    def _load_data(self) -> pd.DataFrame:
        """Load CSV data into DataFrame."""
        return pd.read_csv(self.filepath)

    def _csv_to_dict(self) -> Dict[str, np.ndarray]:
        """Convert CSV columns to dictionary of numpy arrays."""
        return {col: np.array(self.df[col].values[1:]) for col in self.df.columns}

    def split(self):
        # selecting rows based on condition
        rslt_df = self.df[self.df["feature13"] == 5]
        rslt_df = rslt_df[rslt_df["ClaimAmount"] < 20000]
        return rslt_df


def main():
    a = Split("./trainingset.csv")
    splitdf = a.split()
    splitdf.to_csv("feature13=5.csv")


main()
