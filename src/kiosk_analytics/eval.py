"""Evaluation against hand-labelled ground truth.

Ground truth format (CSV): person_id,start_s,end_s
One row per in-zone segment; multiple rows per person are allowed and are
treated as that person's segments under dwell policy (a) (sum of in-zone
segments). The labelling protocol lives in docs/LABELING.md and must
match the pipeline's dwell definition.

Predicted and GT persons are matched one-to-one by Hungarian assignment
on temporal IoU of their engagement windows, then dwell errors are
computed on matched pairs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .analytics import PersonResult


@dataclass
class GtPerson:
    pid: str
    segments: list[tuple[float, float]]

    @property
    def dwell(self) -> float:
        return sum(e - s for s, e in self.segments)


def load_gt(path: str | Path) -> list[GtPerson]:
    persons: dict[str, list[tuple[float, float]]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s, e = float(row["start_s"]), float(row["end_s"])
            if e <= s:
                raise ValueError(f"GT segment with end <= start: {row}")
            persons.setdefault(str(row["person_id"]).strip(), []).append((s, e))
    return [GtPerson(pid, sorted(segs)) for pid, segs in persons.items()]


def _union_length(segments: list[tuple[float, float]]) -> float:
    total, prev_end = 0.0, -np.inf
    for s, e in sorted(segments):
        s = max(s, prev_end)
        if e > s:
            total += e - s
            prev_end = e
    return total


def temporal_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    inter = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            inter += max(0.0, min(e1, e2) - max(s1, s2))
    union = _union_length(list(a) + list(b))
    return inter / union if union > 0 else 0.0


def evaluate(pred: list[PersonResult], gt: list[GtPerson], min_iou: float = 0.1) -> dict:
    pred = [p for p in pred if p.engaged]
    n_pred, n_gt = len(pred), len(gt)

    iou = np.zeros((n_pred, n_gt))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            iou[i, j] = temporal_iou(p.segments, g.segments)
    matches: list[tuple[int, int]] = []
    if n_pred and n_gt:
        rows, cols = linear_sum_assignment(1.0 - iou)
        matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= min_iou]

    per_person = []
    abs_errs, pct_errs = [], []
    for r, c in matches:
        p, g = pred[r], gt[c]
        err = p.dwell_s - g.dwell
        abs_errs.append(abs(err))
        if g.dwell > 0:
            pct_errs.append(abs(err) / g.dwell)
        per_person.append(
            {
                "gt_person": g.pid,
                "pred_person": p.pid,
                "iou": round(float(iou[r, c]), 3),
                "gt_dwell_s": round(g.dwell, 2),
                "pred_dwell_s": p.dwell_s,
                "error_s": round(err, 2),
            }
        )

    matched_pred = {r for r, _ in matches}
    matched_gt = {c for _, c in matches}
    return {
        "count": {
            "gt": n_gt,
            "pred": n_pred,
            "abs_error": abs(n_pred - n_gt),
            "pct_error": round(abs(n_pred - n_gt) / n_gt * 100, 1) if n_gt else None,
        },
        "dwell": {
            "mae_s": round(float(np.mean(abs_errs)), 2) if abs_errs else None,
            "mape_pct": round(float(np.mean(pct_errs)) * 100, 1) if pct_errs else None,
            "n_matched": len(matches),
        },
        "unmatched_pred_persons": [pred[i].pid for i in range(n_pred) if i not in matched_pred],
        "unmatched_gt_persons": [gt[j].pid for j in range(n_gt) if j not in matched_gt],
        "per_person": per_person,
    }
