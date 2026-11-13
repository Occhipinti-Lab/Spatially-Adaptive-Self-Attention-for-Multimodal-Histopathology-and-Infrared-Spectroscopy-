"""Two-phase dataset construction shared by both split strategies."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import savgol_filter

if __package__:
    from .config import ExperimentConfig, LABEL_MAP, REVERSE_LABEL_MAP
    from .preprocessing import SpectralPreprocessor
    from .splitting import patient_id_from_filename, stratified_group_split, stratified_patch_split
else:  # Flat portable deployment: all modules beside main.py.
    from config import ExperimentConfig, LABEL_MAP, REVERSE_LABEL_MAP
    from preprocessing import SpectralPreprocessor
    from splitting import patient_id_from_filename, stratified_group_split, stratified_patch_split

LOGGER = logging.getLogger(__name__)
SPLIT_NAMES = ("train", "validation", "test")
PATCH_SCHEMA = "early_fusion_rgb_spectra_v1"


@dataclass(frozen=True)
class PatchDescriptor:
    uid: str
    filename: str
    patient_id: str
    row: int
    col: int
    original_label: int
    target: int
    row_indices: Tuple[int, ...]


@dataclass
class Partition:
    features: np.ndarray
    targets: np.ndarray
    patch_ids: np.ndarray
    patient_ids: np.ndarray
    filenames: np.ndarray


@dataclass
class DatasetBundle:
    train: Partition
    validation: Partition
    test: Partition
    spectral_channels: int
    feature_names: List[str]
    preprocessor: Optional[SpectralPreprocessor]
    manifest_path: Path
    dataset_fingerprint: str
    split_strategy: str


def _spectral_columns(frame: pd.DataFrame):
    pairs = []
    for column in frame.columns:
        if column.startswith("wn_"):
            try:
                pairs.append((float(column.split("_", 1)[1]), column))
            except ValueError as exc:
                raise ValueError(f"invalid spectral column {column!r}") from exc
    pairs.sort()
    if not pairs:
        raise ValueError("no wn_* spectral columns were found")
    return np.asarray([value for value, _ in pairs]), [name for _, name in pairs]


def load_metadata(path: Path):
    frame = pd.read_csv(path, usecols=["Pixel_Y", "Pixel_X", "Label"])
    coords = frame[["Pixel_Y", "Pixel_X"]].to_numpy()
    if not np.all(np.isfinite(coords)) or np.any(coords < 0) or np.any(coords != coords.astype(int)):
        raise ValueError(f"coordinates must be finite non-negative integers in {path}")
    coords = coords.astype(np.int32)
    if len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError(f"duplicate spatial coordinates in {path}")
    labels = frame["Label"].map(LABEL_MAP).fillna(-1).to_numpy(dtype=np.int32)
    return labels, coords


def load_filtered_spectra(path: Path, cfg: ExperimentConfig):
    frame = pd.read_csv(path)
    required = {"Pixel_Y", "Pixel_X", "Label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing {sorted(required-set(frame.columns))}")
    wavenumbers, columns = _spectral_columns(frame)
    spectra = frame[columns].to_numpy(dtype=np.float64)
    initial = np.zeros(len(wavenumbers), dtype=bool)
    for low, high in cfg.initial_wavenumber_ranges:
        initial |= (wavenumbers >= low) & (wavenumbers <= high)
    spectra, wavenumbers = spectra[:, initial], wavenumbers[initial]
    if spectra.shape[1] < cfg.savgol_window:
        raise ValueError(
            f"{path} retains {spectra.shape[1]} bands, fewer than Savitzky-Golay "
            f"window {cfg.savgol_window}"
        )
    spectra = savgol_filter(
        spectra,
        window_length=cfg.savgol_window,
        polyorder=cfg.savgol_polyorder,
        deriv=cfg.savgol_derivative,
        axis=1,
    )
    final = np.zeros(len(wavenumbers), dtype=bool)
    for low, high in cfg.final_wavenumber_ranges:
        final |= (wavenumbers >= low) & (wavenumbers <= high)
    spectra, wavenumbers = spectra[:, final], wavenumbers[final]
    labels = frame["Label"].map(LABEL_MAP).fillna(-1).to_numpy(dtype=np.int32)
    coords = frame[["Pixel_Y", "Pixel_X"]].to_numpy(dtype=np.int32)
    if spectra.shape[1] == 0 or not np.all(np.isfinite(spectra)):
        raise ValueError(f"no finite spectral bands remain in {path}")
    return spectra, labels, coords, wavenumbers


def _resolve_image(csv_path: Path, image_dir: Path) -> Path:
    stems = [csv_path.stem]
    prefix = csv_path.stem.split("_", 1)[0]
    if prefix != csv_path.stem:
        stems.append(prefix)
    for stem in stems:
        matches = [
            image_dir / f"{stem}{extension}"
            for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
            if (image_dir / f"{stem}{extension}").exists()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"ambiguous images for {csv_path.name}: {matches}")
    raise FileNotFoundError(f"no registered image found for {csv_path.name} in {image_dir}")


def load_image(csv_path: Path, image_dir: Path):
    image = np.asarray(Image.open(_resolve_image(csv_path, image_dir)).convert("RGB"), dtype=np.float32)
    return image / 255.0


def _descriptors_for_file(path: Path, cfg: ExperimentConfig) -> List[PatchDescriptor]:
    labels, coords = load_metadata(path)
    eligible = np.isin(labels, cfg.relevant_labels)
    selected = np.flatnonzero(eligible)
    if not len(selected):
        return []
    tile = coords[selected] // cfg.patch_size
    keys = np.column_stack((tile, labels[selected]))
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    selected, keys = selected[order], keys[order]
    cuts = np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1
    groups = np.split(selected, cuts)
    patient_id = patient_id_from_filename(path.name, cfg.patient_id_regex)
    result = []
    assigned = np.zeros(len(labels), dtype=np.uint8)
    for indices in groups:
        origin = (coords[indices[0]] // cfg.patch_size) * cfg.patch_size
        original_label = int(labels[indices[0]])
        uid_text = f"{path.name}|{int(origin[0])}|{int(origin[1])}|{original_label}"
        uid = hashlib.sha256(uid_text.encode("utf-8")).hexdigest()[:24]
        result.append(
            PatchDescriptor(
                uid=uid,
                filename=path.name,
                patient_id=patient_id,
                row=int(origin[0]),
                col=int(origin[1]),
                original_label=original_label,
                target=int(cfg.class_mapping[original_label]),
                row_indices=tuple(map(int, indices)),
            )
        )
        assigned[indices] += 1
    if not np.all(assigned[eligible] == 1) or np.any(assigned[~eligible]):
        raise RuntimeError(f"patch coverage invariant failed for {path.name}")
    return result


def inventory(cfg: ExperimentConfig) -> List[PatchDescriptor]:
    files = sorted(cfg.data_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"no CSV files found in {cfg.data_dir}")
    LOGGER.info(
        "Dataset inventory: %d CSV files | mode=%s | patch=%dx%d",
        len(files),
        cfg.mode,
        cfg.patch_size,
        cfg.patch_size,
    )
    descriptors = []
    for file_index, path in enumerate(files, start=1):
        LOGGER.info("[Inventory %d/%d] Reading metadata: %s", file_index, len(files), path.name)
        file_descriptors = _descriptors_for_file(path, cfg)
        descriptors.extend(file_descriptors)
        counts = np.bincount(
            [item.target for item in file_descriptors], minlength=cfg.num_classes
        ).tolist() if file_descriptors else [0] * cfg.num_classes
        LOGGER.info(
            "[Inventory %d/%d] %s -> patches=%d, measured_pixels=%d, target_counts=%s",
            file_index,
            len(files),
            path.name,
            len(file_descriptors),
            sum(len(item.row_indices) for item in file_descriptors),
            counts,
        )
    if not descriptors:
        raise ValueError("no eligible patches found")
    uids = [item.uid for item in descriptors]
    if len(uids) != len(set(uids)):
        raise RuntimeError("duplicate patch identifiers were generated")
    return descriptors


def _split(descriptors: Sequence[PatchDescriptor], cfg: ExperimentConfig, seed: int):
    fractions = (cfg.train_fraction, cfg.validation_fraction, cfg.test_fraction)
    if cfg.split_strategy == "patch":
        return stratified_patch_split(
            [item.uid for item in descriptors], [item.target for item in descriptors], fractions, seed
        )
    patients = sorted({item.patient_id for item in descriptors})
    counts = {patient: np.zeros(cfg.num_classes, dtype=int) for patient in patients}
    for item in descriptors:
        counts[item.patient_id][item.target] += 1
    reserved_files = set(cfg.reserved_test_files)
    known_files = {item.filename for item in descriptors}
    if not reserved_files.issubset(known_files):
        raise ValueError(f"unknown reserved files: {sorted(reserved_files-known_files)}")
    reserved_patients = {
        item.patient_id for item in descriptors if item.filename in reserved_files
    }
    patient_split = stratified_group_split(
        counts,
        fractions,
        seed,
        reserved_test_groups=reserved_patients,
        min_groups_per_class=cfg.min_patients_per_class,
    )
    lookup = {patient: split for split, values in patient_split.items() for patient in values}
    return {
        split: sorted(item.uid for item in descriptors if lookup[item.patient_id] == split)
        for split in SPLIT_NAMES
    }


def _fingerprint(descriptors: Sequence[PatchDescriptor]):
    digest = hashlib.sha256()
    for item in sorted(descriptors, key=lambda value: value.uid):
        digest.update(
            f"{item.uid}|{item.filename}|{item.patient_id}|{item.target}|{len(item.row_indices)}\n".encode()
        )
    return digest.hexdigest()


def _fit_preprocessor(
    descriptors: Sequence[PatchDescriptor], train_uids: set[str], cfg: ExperimentConfig, seed: int
):
    if cfg.modality == "image":
        return None
    grouped: Dict[str, List[PatchDescriptor]] = {}
    for item in descriptors:
        if item.uid in train_uids:
            grouped.setdefault(item.filename, []).append(item)
    spectra_parts, target_parts, group_parts = [], [], []
    reference_wavenumbers = None
    rng = np.random.default_rng(seed)
    per_file_limit = (
        0
        if cfg.spectral_fit_max_samples == 0
        else max(4, int(np.ceil(cfg.spectral_fit_max_samples / max(1, len(grouped)))))
    )
    for filename, items in sorted(grouped.items()):
        LOGGER.info(
            "[Preprocessor %d/%d] Loading training spectra from %s (%d patches)",
            len(spectra_parts) + 1,
            len(grouped),
            filename,
            len(items),
        )
        spectra, _, _, wavenumbers = load_filtered_spectra(cfg.data_dir / filename, cfg)
        if reference_wavenumbers is None:
            reference_wavenumbers = wavenumbers
        elif not np.array_equal(reference_wavenumbers, wavenumbers):
            raise ValueError(f"wavenumber grid differs in {filename}")
        rows, labels, groups = [], [], []
        for item in items:
            rows.extend(item.row_indices)
            labels.extend([item.target] * len(item.row_indices))
            groups.extend([item.patient_id] * len(item.row_indices))
        rows = np.asarray(rows, dtype=int)
        labels = np.asarray(labels, dtype=int)
        groups = np.asarray(groups, dtype=str)
        if per_file_limit and len(rows) > per_file_limit:
            keep = rng.choice(len(rows), per_file_limit, replace=False)
            rows, labels, groups = rows[keep], labels[keep], groups[keep]
        spectra_parts.append(spectra[rows])
        target_parts.append(labels)
        group_parts.append(groups)
    fit_x = np.concatenate(spectra_parts)
    fit_y = np.concatenate(target_parts)
    fit_groups = np.concatenate(group_parts)
    if cfg.spectral_fit_max_samples and len(fit_x) > cfg.spectral_fit_max_samples:
        keep = rng.choice(len(fit_x), cfg.spectral_fit_max_samples, replace=False)
        fit_x, fit_y, fit_groups = fit_x[keep], fit_y[keep], fit_groups[keep]
    processor = SpectralPreprocessor()
    scope = (
        "measured pixels from training patches only"
        if cfg.split_strategy == "patch"
        else "measured pixels from training patients only"
    )
    LOGGER.info(
        "Fitting spectral scaler on %d measured spectra (%d raw bands); scope=%s",
        len(fit_x),
        fit_x.shape[1],
        scope,
    )
    fitted = processor.fit(
        fit_x,
        fit_y,
        reference_wavenumbers,
        training_scope=scope,
        groups=fit_groups,
    )
    LOGGER.info(
        "Spectral preprocessing complete: %d -> %d features",
        fit_x.shape[1],
        len(fitted.feature_names),
    )
    return fitted


def _materialize_file(
    filename: str,
    descriptors: Sequence[PatchDescriptor],
    cfg: ExperimentConfig,
    processor: Optional[SpectralPreprocessor],
):
    path = cfg.data_dir / filename
    LOGGER.info(
        "Processing file %s | modality=%s | assigned_patches=%d",
        filename,
        cfg.modality,
        len(descriptors),
    )
    labels, coords = load_metadata(path)
    image = load_image(path, cfg.image_dir) if cfg.modality in {"image", "both"} else None
    spectral = None
    if cfg.modality in {"spectra", "both"}:
        if processor is None:
            raise RuntimeError(
                f"{cfg.modality} requires a fitted spectral preprocessor, but none was supplied"
            )
        raw, raw_labels, raw_coords, wavenumbers = load_filtered_spectra(path, cfg)
        if not np.array_equal(labels, raw_labels) or not np.array_equal(coords, raw_coords):
            raise RuntimeError(f"metadata changed between reads for {filename}")
        spectral = processor.transform(raw, wavenumbers)
    spectral_channels = 0 if spectral is None else spectral.shape[1]
    feature_channels = (3 if cfg.modality in {"image", "both"} else 0) + spectral_channels
    result = {}
    for item in descriptors:
        indices = np.asarray(item.row_indices, dtype=int)
        local = coords[indices] - np.asarray([item.row, item.col])
        if np.any(local < 0) or np.any(local >= cfg.patch_size):
            raise RuntimeError(f"patch coordinate invariant failed for {item.uid}")
        patch = np.zeros((cfg.patch_size, cfg.patch_size, feature_channels), dtype=np.float32)
        cursor = 0
        if image is not None:
            absolute = coords[indices]
            if np.any(absolute[:, 0] >= image.shape[0]) or np.any(absolute[:, 1] >= image.shape[1]):
                raise ValueError(f"coordinates exceed registered image bounds in {filename}")
            patch[local[:, 0], local[:, 1], :3] = image[absolute[:, 0], absolute[:, 1]]
            cursor = 3
        if spectral is not None:
            patch[local[:, 0], local[:, 1], cursor : cursor + spectral_channels] = spectral[indices]
        result[item.uid] = patch
    target_counts = np.bincount(
        [item.target for item in descriptors], minlength=cfg.num_classes
    ).tolist()
    LOGGER.info(
        "Finished %s | extracted_patches=%d | target_counts=%s | tensor_channels=%d",
        filename,
        len(result),
        target_counts,
        feature_channels,
    )
    return result


def build_dataset(cfg: ExperimentConfig, seed: int, run_dir: Path) -> DatasetBundle:
    """Create a split, fit train-only preprocessing, and then materialize patches."""
    cfg.validate()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("=" * 72)
    LOGGER.info(
        "DATA PREPARATION START | split_strategy=%s | split_seed=%d | ACTIVE_MODALITY=%s "
        "role=%s variant=%s",
        cfg.split_strategy,
        seed,
        cfg.modality,
        cfg.experiment_role,
        cfg.ablation_variant or "not_applicable",
    )
    descriptors = inventory(cfg)
    LOGGER.info("Inventory complete: %d total patches across all files", len(descriptors))
    assignments = _split(descriptors, cfg, seed)
    LOGGER.info(
        "Split assignment complete | train=%d | validation=%d | test=%d",
        len(assignments["train"]),
        len(assignments["validation"]),
        len(assignments["test"]),
    )
    split_for_uid = {uid: split for split, values in assignments.items() for uid in values}
    train_uids = set(assignments["train"])
    processor = _fit_preprocessor(descriptors, train_uids, cfg, seed)
    if cfg.modality in {"spectra", "both"} and processor is None:
        raise RuntimeError("spectral preprocessing was not fitted")
    preprocessing_contract = {
        "active_modality": cfg.modality,
        "spectral_input_active": cfg.modality in {"spectra", "both"},
        "preprocessor_created": processor is not None,
        "training_scope": None if processor is None else processor.training_scope_,
        "status": "verified",
    }
    (run_dir / "preprocessing_contract.json").write_text(
        json.dumps(preprocessing_contract, indent=2), encoding="utf-8"
    )
    if processor is not None:
        processor.save(run_dir / "spectral_preprocessor.pkl")

    by_file: Dict[str, List[PatchDescriptor]] = {}
    for item in descriptors:
        by_file.setdefault(item.filename, []).append(item)
    patch_lookup = {}
    for file_index, (filename, items) in enumerate(sorted(by_file.items()), start=1):
        LOGGER.info("[Extraction %d/%d] Starting %s", file_index, len(by_file), filename)
        patch_lookup.update(_materialize_file(filename, items, cfg, processor))

    manifest_rows = []
    partitions = {}
    spectral_channels = 0 if processor is None else len(processor.feature_names)
    channel_count = (3 if cfg.modality in {"image", "both"} else 0) + spectral_channels
    for split in SPLIT_NAMES:
        items = sorted(
            (item for item in descriptors if split_for_uid[item.uid] == split), key=lambda x: x.uid
        )
        if not items:
            raise ValueError(f"{split} is empty")
        targets = np.asarray([item.target for item in items], dtype=np.int32)
        counts = np.bincount(targets, minlength=cfg.num_classes)
        if np.any(counts < cfg.min_patches_per_class):
            raise ValueError(
                f"{split} has insufficient target patches: {counts.tolist()} "
                f"(minimum {cfg.min_patches_per_class})"
            )
        features = np.stack([patch_lookup[item.uid] for item in items]).astype(np.float32)
        if features.shape[1:] != (cfg.patch_size, cfg.patch_size, channel_count):
            raise RuntimeError("materialized channel schema mismatch")
        partitions[split] = Partition(
            features=features,
            targets=targets,
            patch_ids=np.asarray([item.uid for item in items]),
            patient_ids=np.asarray([item.patient_id for item in items]),
            filenames=np.asarray([item.filename for item in items]),
        )
        LOGGER.info(
            "SPLIT SUMMARY | %-10s samples=%d patients=%d files=%d class_counts=%s shape=%s",
            split,
            len(targets),
            len(np.unique(partitions[split].patient_ids)),
            len(np.unique(partitions[split].filenames)),
            counts.tolist(),
            features.shape,
        )
        for item in items:
            manifest_rows.append(
                {
                    "split": split,
                    "split_strategy": cfg.split_strategy,
                    "split_seed": seed,
                    "patch_uid": item.uid,
                    "patient_id": item.patient_id,
                    "filename": item.filename,
                    "start_row": item.row,
                    "start_col": item.col,
                    "original_label": REVERSE_LABEL_MAP[item.original_label],
                    "model_target": item.target,
                    "measured_pixels": len(item.row_indices),
                }
            )
    manifest_path = run_dir / "split_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    fingerprint = _fingerprint(descriptors)
    audit = {
        "schema": PATCH_SCHEMA,
        "dataset_fingerprint": fingerprint,
        "split_seed": seed,
        "active_modality": cfg.modality,
        "experiment_role": cfg.experiment_role,
        "ablation_variant": cfg.ablation_variant,
        "parent_requested_modality": cfg.parent_requested_modality,
        "split_strategy": cfg.split_strategy,
        "patient_disjoint": cfg.split_strategy == "patient",
        "patch_disjoint": True,
        "preprocessing_scope": None if processor is None else processor.training_scope_,
        "preprocessing_contract": preprocessing_contract,
        "feature_order": (
            ["R", "G", "B"] if cfg.modality in {"image", "both"} else []
        )
        + ([] if processor is None else processor.feature_names),
        "split_counts": {
            name: np.bincount(partitions[name].targets, minlength=cfg.num_classes).tolist()
            for name in SPLIT_NAMES
        },
        "patient_overlap": {
            "train_validation": sorted(
                set(partitions["train"].patient_ids) & set(partitions["validation"].patient_ids)
            ),
            "train_test": sorted(
                set(partitions["train"].patient_ids) & set(partitions["test"].patient_ids)
            ),
            "validation_test": sorted(
                set(partitions["validation"].patient_ids) & set(partitions["test"].patient_ids)
            ),
        },
    }
    (run_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    LOGGER.info(
        "DATA PREPARATION COMPLETE | train=%d | validation=%d | test=%d | manifest=%s",
        len(partitions["train"].targets),
        len(partitions["validation"].targets),
        len(partitions["test"].targets),
        manifest_path,
    )
    LOGGER.info("=" * 72)
    return DatasetBundle(
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        spectral_channels=spectral_channels,
        feature_names=[] if processor is None else processor.feature_names,
        preprocessor=processor,
        manifest_path=manifest_path,
        dataset_fingerprint=fingerprint,
        split_strategy=cfg.split_strategy,
    )
