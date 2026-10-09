"""Cross-validation split construction."""

from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

Split = Tuple[np.ndarray, np.ndarray]


def stratified_kfold(labels: Sequence[int], n_splits: int, seed: int) -> List[Split]:
    """Image-level stratified k-fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = np.asarray(labels)
    return list(skf.split(np.zeros(len(labels)), labels))


def build_group_ids(df: pd.DataFrame, min_chars: int = 25) -> pd.Series:
    """Group images that are very likely slices from the same patient/report.

    Key = (contributor directory, sorted matched-span text). Images sharing an
    identical extraction within one directory share a report. Spans shorter
    than `min_chars` (e.g. a lone "thickening") are too generic to group on.
    """
    directory = df["relative_path"].apply(os.path.dirname)

    def span_text(s) -> str:
        try:
            d = json.loads(s) if isinstance(s, str) and s.strip() else {}
        except Exception:
            return ""
        vals = []
        for v in d.values():
            if isinstance(v, list):
                vals.extend(str(x).strip().lower() for x in v)
        return " | ".join(sorted(vals))

    spans = df["matched_spans"].fillna("").apply(span_text)
    keys = list(zip(directory, spans))
    counts = pd.Series(keys).value_counts()

    group_id, key_to_gid, next_gid = [None] * len(df), {}, 0
    for i, key in enumerate(keys):
        if len(key[1]) < min_chars or counts[key] < 2:
            group_id[i] = next_gid
            next_gid += 1
        else:
            if key not in key_to_gid:
                key_to_gid[key] = next_gid
                next_gid += 1
            group_id[i] = key_to_gid[key]
    return pd.Series(group_id, index=df.index, name="group_id")


def grouped_multilabel_kfold(
    df: pd.DataFrame, label_cols: Sequence[str], n_splits: int, seed: int, min_chars: int = 25, log=print
) -> Tuple[List[Split], pd.Series]:
    """Group-respecting MultilabelStratifiedKFold.

    Stratifies on group-level label vectors (max over member rows) and expands
    the group assignment back to rows, so no group spans train and val.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    group_ids = build_group_ids(df, min_chars=min_chars)
    rows_by_group = {}
    for row_idx, gid in enumerate(group_ids):
        rows_by_group.setdefault(gid, []).append(row_idx)
    unique_gids = sorted(rows_by_group)

    group_labels = np.stack([df.iloc[rows_by_group[g]][list(label_cols)].values.max(axis=0) for g in unique_gids])
    group_x = np.arange(len(unique_gids)).reshape(-1, 1)

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for tr_g, va_g in mskf.split(group_x, group_labels):
        tr_idx = np.sort(np.concatenate([rows_by_group[unique_gids[g]] for g in tr_g]))
        va_idx = np.sort(np.concatenate([rows_by_group[unique_gids[g]] for g in va_g]))
        splits.append((tr_idx, va_idx))

    multi = [g for g in unique_gids if len(rows_by_group[g]) > 1]
    n_multi_imgs = sum(len(rows_by_group[g]) for g in multi)
    log(f"Grouped split: {len(df)} images -> {len(unique_gids)} groups ({len(multi)} multi-image groups "
        f"covering {n_multi_imgs} images, {100 * n_multi_imgs / len(df):.1f}%) -- these never span train/val")
    return splits, group_ids
