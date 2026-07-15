"""Spectral extraction and pathology annotation tools."""

from .agilent_reader import AgilentIRFile
from .annotation import (
    DEFAULT_LABEL_COLORS,
    AnnotationSummary,
    BatchAnnotationSummary,
    label_spectral_csv,
    label_spectral_directory,
)
from .spectral_export import (
    SpectralExportSummary,
    discover_agilent_files,
    extract_spectral_file,
    extract_spectral_files,
)

__all__ = [
    "AgilentIRFile",
    "AnnotationSummary",
    "BatchAnnotationSummary",
    "DEFAULT_LABEL_COLORS",
    "SpectralExportSummary",
    "discover_agilent_files",
    "extract_spectral_file",
    "extract_spectral_files",
    "label_spectral_csv",
    "label_spectral_directory",
]

__version__ = "0.1.0"
