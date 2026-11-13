"""Deterministic patch- and patient-level partitioning with hard audits."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "validation", "test")


def patient_id_from_filename(filename: str, pattern: str | None = None) -> str:
    stem = Path(filename).stem
    if pattern:
        match = re.search(pattern, stem)
        if match is None:
            raise ValueError(f"{filename!r} does not match patient regex {pattern!r}")
        patient_id = match.group(1) if match.groups() else match.group(0)
    else:
        patient_id = stem.split("_", 1)[0]
    if not patient_id:
        raise ValueError(f"could not infer patient ID from {filename!r}")
    return patient_id


def _capacities(total: int, fractions: Sequence[float]) -> np.ndarray:
    exact = np.asarray(fractions, dtype=float) * total
    result = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - result))[: total - result.sum()]:
        result[index] += 1
    if total < 3 or np.any(result == 0):
        raise ValueError("at least three groups are required and each split must be non-empty")
    return result


def audit_partitions(
    assignments: Mapping[str, Sequence[str]], expected: Iterable[str], unit: str
) -> None:
    membership = defaultdict(list)
    for split in SPLIT_NAMES:
        if split not in assignments:
            raise ValueError(f"missing split {split!r}")
        for item in assignments[split]:
            membership[str(item)].append(split)
    duplicates = {item: splits for item, splits in membership.items() if len(splits) != 1}
    if duplicates:
        raise RuntimeError(f"{unit} leakage across splits: {duplicates}")
    expected_set = {str(value) for value in expected}
    if set(membership) != expected_set:
        raise RuntimeError(
            f"{unit} partition mismatch; missing={sorted(expected_set-set(membership))}, "
            f"unexpected={sorted(set(membership)-expected_set)}"
        )


def stratified_patch_split(
    patch_ids: Sequence[str], targets: Sequence[int], fractions, seed: int
) -> Dict[str, list[str]]:
    ids = np.asarray(patch_ids, dtype=object)
    y = np.asarray(targets, dtype=int)
    if len(ids) != len(set(ids.tolist())):
        raise ValueError("patch IDs must be unique")
    try:
        train_ids, remainder, train_y, remainder_y = train_test_split(
            ids,
            y,
            train_size=float(fractions[0]),
            random_state=int(seed),
            stratify=y,
        )
        relative_validation = float(fractions[1]) / (float(fractions[1]) + float(fractions[2]))
        validation_ids, test_ids = train_test_split(
            remainder,
            train_size=relative_validation,
            random_state=int(seed) + 1,
            stratify=remainder_y,
        )
    except ValueError as exc:
        raise ValueError(
            "patch stratification failed; each target needs enough patches for all splits"
        ) from exc
    result = {
        "train": sorted(map(str, train_ids)),
        "validation": sorted(map(str, validation_ids)),
        "test": sorted(map(str, test_ids)),
    }
    audit_partitions(result, ids, "patch")
    return result


def _allocation_score(allocation, group_ids, counts, capacities) -> float:
    index = {group_id: i for i, group_id in enumerate(group_ids)}
    totals = np.maximum(counts.sum(axis=0), 1)
    target = capacities / capacities.sum()
    score = 0.0
    for split_index, split in enumerate(SPLIT_NAMES):
        rows = [index[group_id] for group_id in allocation[split]]
        observed = counts[rows].sum(axis=0) / totals if rows else np.zeros(counts.shape[1])
        score += float(np.square(observed - target[split_index]).sum())
    return score


def _feasible_milp_allocation(
    group_ids, counts, capacities, reserved, min_groups_per_class, seed
):

    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    group_count, class_count = counts.shape
    row_count = group_count + 3 + 3 * class_count
    constraints = lil_matrix((row_count, 3 * group_count), dtype=float)
    lower = np.full(row_count, -np.inf)
    upper = np.full(row_count, np.inf)
    for group_index in range(group_count):
        constraints[group_index, 3 * group_index : 3 * group_index + 3] = 1
        lower[group_index] = upper[group_index] = 1
    offset = group_count
    for split_index in range(3):
        columns = np.arange(split_index, 3 * group_count, 3)
        constraints[offset + split_index, columns] = 1
        lower[offset + split_index] = upper[offset + split_index] = capacities[split_index]
        for class_index in range(class_count):
            row = offset + 3 + split_index * class_count + class_index
            constraints[row, columns] = (counts[:, class_index] > 0).astype(float)
            lower[row] = min_groups_per_class
    lower_bounds = np.zeros(3 * group_count)
    upper_bounds = np.ones(3 * group_count)
    for group_index, group_id in enumerate(group_ids):
        if group_id in reserved:
            upper_bounds[3 * group_index : 3 * group_index + 2] = 0
            lower_bounds[3 * group_index + 2] = 1
    rng = np.random.default_rng(seed)
    objective = rng.uniform(0.0, 1e-4, size=3 * group_count)
    result = milp(
        objective,
        integrality=np.ones(3 * group_count),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(constraints.tocsc(), lower, upper),
        options={"time_limit": 30.0},
    )
    if result.x is None:
        raise ValueError(
            "no patient allocation satisfies split capacities, reservations, and class coverage"
        )
    decisions = np.rint(result.x.reshape(group_count, 3)).astype(int)
    if not np.all(decisions.sum(axis=1) == 1):
        raise RuntimeError("patient split solver returned a non-integral assignment")
    return {
        name: [group_ids[index] for index in np.flatnonzero(decisions[:, split_index])]
        for split_index, name in enumerate(SPLIT_NAMES)
    }


def stratified_group_split(
    group_class_counts: Mapping[str, Sequence[int]],
    fractions,
    seed: int,
    reserved_test_groups: Iterable[str] = (),
    min_groups_per_class: int = 1,
    trials: int = 2048,
) -> Dict[str, list[str]]:

    group_ids = sorted(map(str, group_class_counts))
    counts = np.vstack([np.asarray(group_class_counts[group], dtype=int) for group in group_ids])
    if np.any(counts < 0) or counts.ndim != 2:
        raise ValueError("group class counts must be a non-negative matrix")
    prevalence = (counts > 0).sum(axis=0)
    required = 3 * int(min_groups_per_class)
    if np.any(prevalence < required):
        bad = {int(i): int(prevalence[i]) for i in np.flatnonzero(prevalence < required)}
        raise ValueError(f"each class needs at least {required} patient groups; observed {bad}")
    reserved = set(map(str, reserved_test_groups))
    if not reserved.issubset(group_ids):
        raise ValueError(f"unknown reserved groups: {sorted(reserved-set(group_ids))}")
    capacities = _capacities(len(group_ids), fractions)
    if len(reserved) > capacities[2]:
        raise ValueError("reserved test patients exceed the requested test capacity")

    movable = np.asarray([group for group in group_ids if group not in reserved], dtype=object)
    open_capacities = capacities.copy()
    open_capacities[2] -= len(reserved)
    rng = np.random.default_rng(seed)
    best = None
    best_score = np.inf
    lookup = {group: i for i, group in enumerate(group_ids)}
    for _ in range(max(1, int(trials))):
        shuffled = movable[rng.permutation(len(movable))]
        a, b = open_capacities[0], open_capacities[0] + open_capacities[1]
        proposal = {
            "train": shuffled[:a].tolist(),
            "validation": shuffled[a:b].tolist(),
            "test": sorted(reserved) + shuffled[b:].tolist(),
        }
        coverage_ok = all(
            np.all((counts[[lookup[g] for g in proposal[split]]] > 0).sum(axis=0) >= min_groups_per_class)
            for split in SPLIT_NAMES
        )
        if not coverage_ok:
            continue
        score = _allocation_score(proposal, group_ids, counts, capacities)
        if score < best_score:
            best, best_score = proposal, score
    if best is None:
        best = _feasible_milp_allocation(
            group_ids,
            counts,
            capacities,
            reserved,
            int(min_groups_per_class),
            seed,
        )
    result = {split: sorted(map(str, best[split])) for split in SPLIT_NAMES}
    audit_partitions(result, group_ids, "patient")
    return result
