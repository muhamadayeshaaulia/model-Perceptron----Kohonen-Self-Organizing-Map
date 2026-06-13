import pandas as pd
import numpy as np
from utils.helper import normalisasi


class DatasetRepository:
    """Bertanggung jawab untuk akses dan persiapan data dari dataset.csv."""

    FITUR = ["jam_tidur", "mood", "stres", "jam_belajar", "jam_hp", "jumlah_tugas"]
    CSV_PATH = "dataset.csv"

    def __init__(self):
        self._df: pd.DataFrame = pd.DataFrame()
        self._X: np.ndarray = np.empty((0, 0))
        self._y: np.ndarray = np.empty(0)
        self._X_norm: np.ndarray = np.empty((0, 0))
        self._nilai_min: np.ndarray = np.empty(0)
        self._nilai_max: np.ndarray = np.empty(0)

    def load(self):
        """Load dataset dari CSV dan siapkan fitur, label, dan data ternormalisasi."""
        self._df = pd.read_csv(self.CSV_PATH)
        self._X = self._df[self.FITUR].values
        self._y = self._df["label"].values
        self._X_norm, self._nilai_min, self._nilai_max = normalisasi(self._X)
        return self

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def X(self) -> np.ndarray:
        return self._X

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def X_norm(self) -> np.ndarray:
        return self._X_norm

    @property
    def nilai_min(self) -> np.ndarray:
        return self._nilai_min

    @property
    def nilai_max(self) -> np.ndarray:
        return self._nilai_max

    @property
    def fitur(self) -> list:
        return self.FITUR
