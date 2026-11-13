"""Static cross-patient duplicate audit performed before any model selection."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

if __package__:
    from .config import ExperimentConfig
    from .data import _resolve_image
    from .splitting import patient_id_from_filename
else:  # Flat portable deployment: all modules beside main.py.
    from config import ExperimentConfig
    from data import _resolve_image
    from splitting import patient_id_from_filename

LOGGER = logging.getLogger(__name__)


def _file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _measurement_hash(frame: pd.DataFrame):

    spectral = sorted(
        (column for column in frame if column.startswith("wn_")),
        key=lambda value: float(value.split("_", 1)[1]),
    )
    if not spectral:
        raise ValueError("no wn_* spectral columns found")
    columns = ["Pixel_Y", "Pixel_X"] + spectral
    numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    row_hashes = pd.util.hash_pandas_object(numeric, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256("|".join(columns).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _cross_patient(records, key):
    grouped = defaultdict(list)
    for record in records:
        grouped[record[key]].append(record)
    collisions = []
    for signature, values in grouped.items():
        patients = sorted({value["patient_id"] for value in values})
        if len(patients) > 1:
            collisions.append(
                {
                    "signature": signature,
                    "patients": patients,
                    "files": sorted({value["filename"] for value in values}),
                }
            )
    return collisions


def run_duplicate_audit(cfg: ExperimentConfig, output_path: Path, strict: bool = True):
    records = []
    csv_paths = sorted(cfg.data_dir.glob("*.csv"))
    if not csv_paths:
        raise ValueError(f"no CSV files found for duplicate audit in {cfg.data_dir}")
    LOGGER.info("Static dataset audit: %d CSV files", len(csv_paths))
    for file_index, path in enumerate(csv_paths, start=1):
        LOGGER.info("[Audit %d/%d] Checking %s", file_index, len(csv_paths), path.name)
        image = (
            _resolve_image(path, cfg.image_dir)
            if cfg.modality in {"image", "both"}
            else None
        )
        frame = pd.read_csv(path)
        required = {"Pixel_Y", "Pixel_X", "Label"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} is missing {sorted(required-set(frame.columns))}")
        coords = frame[["Pixel_Y", "Pixel_X"]].apply(pd.to_numeric, errors="raise").to_numpy()
        if not np.all(np.isfinite(coords)) or np.any(coords < 0) or np.any(coords != np.rint(coords)):
            raise ValueError(f"invalid coordinates in {path}")
        if len(np.unique(coords.astype(np.int64), axis=0)) != len(coords):
            raise ValueError(f"duplicate coordinates in {path}")
        if image is not None:
            with Image.open(image) as opened:
                width, height = opened.size
            if len(coords) and (
                coords[:, 0].max() >= height or coords[:, 1].max() >= width
            ):
                raise ValueError(
                    f"coordinates in {path.name} exceed image bounds {(height, width)}"
                )
        record = {
            "filename": path.name,
            "patient_id": patient_id_from_filename(path.name, cfg.patient_id_regex),
            "csv_sha256": _file_hash(path),
            "measurement_without_label_sha256": _measurement_hash(frame),
            "image_path": None if image is None else str(image),
            "image_sha256": None if image is None else _file_hash(image),
        }
        records.append(record)
    collisions = {
        "identical_csv_across_patients": _cross_patient(records, "csv_sha256"),
        "identical_measurements_without_label_across_patients": _cross_patient(
            records, "measurement_without_label_sha256"
        ),
        "identical_image_across_patients": (
            _cross_patient(records, "image_sha256")
            if cfg.modality in {"image", "both"}
            else []
        ),
    }
    report = {
        "files_audited": len(records),
        "patient_id_rule": cfg.patient_id_regex or "prefix before first underscore",
        "active_modality": cfg.modality,
        "image_audit_applicable": cfg.modality in {"image", "both"},
        "collisions": collisions,
        "passed": not any(collisions.values()),
        "records": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if strict and not report["passed"]:
        summary = {name: len(value) for name, value in collisions.items() if value}
        raise RuntimeError(
            f"cross-patient duplicate audit failed: {summary}; inspect {output_path}"
        )
    LOGGER.info(
        "Static dataset audit complete: files=%d, passed=%s, report=%s",
        len(records),
        report["passed"],
        output_path,
    )
    return report
