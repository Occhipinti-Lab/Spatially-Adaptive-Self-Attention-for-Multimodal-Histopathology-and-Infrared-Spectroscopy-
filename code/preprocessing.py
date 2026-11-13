
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler


class SpectralPreprocessor:
    def __init__(self):
        self.input_scaler = RobustScaler(quantile_range=(5.0, 95.0))
        self.wavenumbers_ = None
        self.feature_names_ = None
        self.training_scope_ = None

    def fit(self, spectra, labels, wavenumbers, training_scope=None, groups=None):
        spectra = np.asarray(spectra, dtype=np.float64)
        wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
        if spectra.ndim != 2 or spectra.shape[0] < 4 or spectra.shape[1] == 0:
            raise ValueError(f"training spectra must be NxC with N>=4; got {spectra.shape}")
        if wavenumbers.shape != (spectra.shape[1],):
            raise ValueError("wavenumber vector does not match spectral columns")
        if not np.all(np.isfinite(spectra)):
            raise ValueError("training spectra must be finite")
        self.input_scaler.fit(spectra)
        self.wavenumbers_ = wavenumbers.copy()
        self.feature_names_ = [f"WN_{value:.2f}" for value in wavenumbers]
        self.training_scope_ = training_scope
        return self

    def transform(self, spectra, wavenumbers):
        if self.wavenumbers_ is None:
            raise RuntimeError("spectral preprocessor has not been fitted")
        wavenumbers = np.asarray(wavenumbers, dtype=np.float64)
        if wavenumbers.shape != self.wavenumbers_.shape or not np.allclose(wavenumbers, self.wavenumbers_):
            raise ValueError("spectral wavenumbers differ from the training set")
        spectra = np.asarray(spectra, dtype=np.float64)
        if spectra.ndim != 2 or spectra.shape[1] != len(self.wavenumbers_):
            raise ValueError("unexpected spectral input shape")
        if not np.all(np.isfinite(spectra)):
            raise ValueError("spectra must be finite")
        return self.input_scaler.transform(spectra).astype(np.float32)

    @property
    def feature_names(self):
        if self.feature_names_ is None:
            raise RuntimeError("spectral preprocessor has not been fitted")
        return list(self.feature_names_)

    def save(self, path: Path):
        with Path(path).open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        with Path(path).open("rb") as file:
            loaded = pickle.load(file)
        if not isinstance(loaded, cls):
            raise TypeError("incorrect spectral preprocessing artifact")
        return loaded
