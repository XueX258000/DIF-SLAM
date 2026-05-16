#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal unit tests for FastSAM "everything" post-processing.

Scope (per `docs/参考资料/DIF_SLAM_工程实现方案.md`):
- label_map mutual exclusion (overlap resolution)
- border-touch filtering and area gating

Run:
  conda activate pslam38
  python scripts/dif/test_fastsam_postprocess.py
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np


def _import_server():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.dif.fastsam_server import (  # type: ignore
        PostprocessConfig,
        _build_label_map,
        _postprocess_masks,
    )

    return PostprocessConfig, _build_label_map, _postprocess_masks


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_build_label_map_overlap_resolution() -> None:
    PostprocessConfig, _build_label_map, _postprocess_masks = _import_server()
    _ = (PostprocessConfig, _postprocess_masks)

    h, w = 10, 12
    m1 = np.zeros((h, w), dtype=bool)
    m2 = np.zeros((h, w), dtype=bool)
    m1[:, 2:8] = True
    m2[:, 6:11] = True  # overlap cols [6,7]

    label = _build_label_map([m1, m2], [0.10, 0.90], h, w)
    _assert(label.dtype == np.int16, "label_map dtype must be int16")

    # Overlap should belong to higher-score mask (m2). It becomes inst_id=0 due to score ordering.
    overlap = m1 & m2
    _assert(int(np.any(overlap)) == 1, "test setup error: no overlap")
    _assert(np.all(label[overlap] == 0), "overlap pixels must be assigned to higher-score mask")

    # Mutual exclusivity: reconstructed instance masks should be disjoint.
    max_id = int(label.max())
    masks: List[np.ndarray] = []
    for inst_id in range(max_id + 1):
        masks.append(label == inst_id)
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            _assert(not bool(np.any(masks[i] & masks[j])), "instances must be mutually exclusive")


def test_postprocess_border_touch_filter() -> None:
    PostprocessConfig, _build_label_map, _postprocess_masks = _import_server()
    _ = _build_label_map

    h, w = 20, 30
    full = np.ones((h, w), dtype=bool)  # touches 4 borders
    inner = np.zeros((h, w), dtype=bool)
    inner[5:15, 10:20] = True  # touches 0 borders

    cfg = PostprocessConfig(
        area_min=10,
        area_max_ratio=1.0,
        border_touch_min_sides=3,
        morph_kernel=0,
        iou_nms=0.99,
        max_masks=50,
    )
    masks, scores = _postprocess_masks([full, inner], [1.0, 1.0], cfg)
    _assert(len(masks) == 1, "border-touch filter should drop background-like full mask")
    _assert(int(masks[0].sum()) == int(inner.sum()), "remaining mask must be the inner one")


def test_postprocess_area_max_ratio_filter() -> None:
    PostprocessConfig, _build_label_map, _postprocess_masks = _import_server()
    _ = _build_label_map

    h, w = 20, 20
    big = np.zeros((h, w), dtype=bool)
    big[:, :] = True  # 100% area
    small = np.zeros((h, w), dtype=bool)
    small[0:5, 0:5] = True  # 6.25% area

    cfg = PostprocessConfig(
        area_min=10,
        area_max_ratio=0.65,
        border_touch_min_sides=0,
        morph_kernel=0,
        iou_nms=0.99,
        max_masks=50,
    )
    masks, scores = _postprocess_masks([big, small], [1.0, 1.0], cfg)
    _assert(len(masks) == 1, "area_max_ratio filter should drop overly-large mask")
    _assert(int(masks[0].sum()) == int(small.sum()), "remaining mask must be the small one")


def main() -> int:
    test_build_label_map_overlap_resolution()
    test_postprocess_border_touch_filter()
    test_postprocess_area_max_ratio_filter()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
