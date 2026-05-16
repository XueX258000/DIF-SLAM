#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import io
import json
import os
import random
import struct
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from fastsam import FastSAM
from fastsam.predict import FastSAMPredictor

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _default_everything_vis_dir() -> str:
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    except Exception:
        root = os.getcwd()
    return os.path.join(root, "Output", "everything_mask")


def _default_post_processing_vis_dir() -> str:
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    except Exception:
        root = os.getcwd()
    return os.path.join(root, "Output", "post_processing")


def _color_for_mask_index(idx: int) -> Tuple[int, int, int]:
    # Deterministic vivid palette in HSV, mapped to BGR for OpenCV drawing.
    hue = int((idx * 37) % 180)
    hsv = np.uint8([[[hue, 230, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _render_everything_raw_vis(
    base_bgr: np.ndarray,
    masks: List[np.ndarray],
    alpha: float = 0.45,
) -> np.ndarray:
    if base_bgr is None or base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError("invalid base image")
    if not masks:
        return base_bgr

    out_f = base_bgr.astype(np.float32, copy=True)
    a = float(np.clip(alpha, 0.0, 1.0))

    for i, m in enumerate(masks):
        if m is None:
            continue
        if m.dtype != np.bool_:
            m = m.astype(np.bool_)
        if m.shape[0] != base_bgr.shape[0] or m.shape[1] != base_bgr.shape[1]:
            continue
        if not bool(m.any()):
            continue
        color = np.array(_color_for_mask_index(i), dtype=np.float32)
        out_f[m] = out_f[m] * (1.0 - a) + color * a

    return np.clip(out_f, 0.0, 255.0).astype(np.uint8)


def _render_post_processing_vis(
    base_bgr: np.ndarray,
    label_map: np.ndarray,
    alpha: float = 0.35,
    contour_thickness: int = 2,
) -> np.ndarray:
    if base_bgr is None or base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError("invalid base image")
    if label_map is None or label_map.ndim != 2:
        raise ValueError("invalid label_map")
    if label_map.shape[0] != base_bgr.shape[0] or label_map.shape[1] != base_bgr.shape[1]:
        raise ValueError("label_map size mismatch")

    out_f = base_bgr.astype(np.float32, copy=True)
    a = float(np.clip(alpha, 0.0, 1.0))

    labels = np.unique(label_map)
    labels = labels[labels >= 0]
    if labels.size == 0:
        return base_bgr
    labels = np.sort(labels.astype(np.int32, copy=False))

    k = np.ones((3, 3), np.uint8)
    thick = int(max(0, contour_thickness))

    for lab in labels.tolist():
        mask = (label_map == int(lab))
        if not bool(mask.any()):
            continue

        color = np.array(_color_for_mask_index(int(lab)), dtype=np.float32)
        out_f[mask] = out_f[mask] * (1.0 - a) + color * a

        if thick > 0:
            m_u8 = (mask.astype(np.uint8) * 255)
            # 1px-ish inner boundary: edge = mask - erode(mask).
            er = cv2.erode(m_u8, k, iterations=1)
            edge = cv2.subtract(m_u8, er)
            if thick > 1:
                edge = cv2.dilate(edge, k, iterations=thick - 1)
            edge_mask = edge > 0
            # Use a dark outline to match the DIF visualization style (without any text).
            out_f[edge_mask] = 0.0

    return np.clip(out_f, 0.0, 255.0).astype(np.uint8)


def _configure_torch_runtime(deterministic: bool, seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    try:
        torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = not bool(deterministic)
            torch.backends.cudnn.deterministic = bool(deterministic)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    # Ensure stable cublas workspace config when possible (takes effect only if set early).
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise EOFError("unexpected EOF")
        buf.extend(chunk)
    return bytes(buf)


def _read_message(stream) -> Tuple[Dict[str, Any], bytes]:
    header_len_bytes = stream.read(4)
    if not header_len_bytes:
        raise EOFError("EOF")
    if len(header_len_bytes) != 4:
        raise EOFError("truncated header length")
    (header_len,) = struct.unpack(">I", header_len_bytes)
    header = json.loads(_read_exact(stream, header_len).decode("utf-8"))
    payload_len = int(header.get("payload_len", 0))
    payload = _read_exact(stream, payload_len) if payload_len > 0 else b""
    return header, payload


def _write_message(stream, header: Dict[str, Any], payload: bytes) -> None:
    header = dict(header)
    header["payload_len"] = int(len(payload))
    header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    try:
        stream.write(struct.pack(">I", len(header_bytes)))
        stream.write(header_bytes)
        if payload:
            stream.write(payload)
        stream.flush()
    except BrokenPipeError:
        # Parent process closed the pipe (normal during shutdown). Exit quietly.
        raise


@dataclass
class PostprocessConfig:
    area_min: int
    area_max_ratio: float
    border_touch_min_sides: int
    morph_kernel: int
    iou_nms: float
    max_masks: int
    # Connected-components cleanup inside each mask (0 disables).
    cc_min_area: int = 0
    # Overlap-resolution assignment thresholds:
    # Only accept a new instance label if it contributes enough free pixels.
    min_assign_pixels: int = 0
    min_assign_ratio: float = 0.0
    # Global cleanup on final label_map: remove/merge tiny islands (0 disables).
    island_min_area: int = 0


def _filter_mask_connected_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_)
    m_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
    if num <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_ids = np.where(areas >= int(min_area))[0] + 1
    if keep_ids.size == 0:
        keep_ids = np.array([int(np.argmax(areas)) + 1], dtype=np.int32)
    keep = np.isin(labels, keep_ids)
    return keep.astype(np.bool_)


def _ann_to_masks_and_scores(ann: Any, h: int, w: int) -> Tuple[List[np.ndarray], List[float]]:
    masks: List[np.ndarray] = []
    scores: List[float] = []

    def _as_numpy(x: Any) -> np.ndarray:
        if torch is not None and isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    if isinstance(ann, (list, tuple)) and len(ann) > 0:
        if isinstance(ann[0], dict) and ("segmentation" in ann[0]):
            for a in ann:
                m = _as_numpy(a["segmentation"])
                s = float(a.get("score", 1.0))
                masks.append(m)
                scores.append(s)
        else:
            for a in ann:
                masks.append(_as_numpy(a))
                scores.append(1.0)
    elif hasattr(ann, "ndim") and getattr(ann, "ndim") == 3:
        for i in range(int(ann.shape[0])):
            masks.append(_as_numpy(ann[i]))
            scores.append(1.0)
    else:
        return [], []

    out_masks: List[np.ndarray] = []
    out_scores: List[float] = []
    for m, s in zip(masks, scores):
        m = _as_numpy(m)
        if m.dtype != np.bool_:
            m = m.astype(np.bool_)
        if m.ndim == 2:
            pass
        elif m.ndim == 3:
            m = m.squeeze()
        else:
            continue

        if m.shape[0] != h or m.shape[1] != w:
            m_u8 = (m.astype(np.uint8) * 255)
            m_u8 = cv2.resize(m_u8, (w, h), interpolation=cv2.INTER_NEAREST)
            m = m_u8 > 0

        out_masks.append(m)
        out_scores.append(float(s))

    return out_masks, out_scores


def _mask_bbox_xyxy(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return x1, y1, x2, y2


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = float(iw * ih)
    area_a = float((ax2 - ax1 + 1) * (ay2 - ay1 + 1))
    area_b = float((bx2 - bx1 + 1) * (by2 - by1 + 1))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    if inter <= 0:
        return 0.0
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def _mask_iou_from_bbox(
    a: np.ndarray,
    area_a: int,
    bbox_a: Tuple[int, int, int, int],
    b: np.ndarray,
    area_b: int,
    bbox_b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = int(np.logical_and(a[iy1 : iy2 + 1, ix1 : ix2 + 1], b[iy1 : iy2 + 1, ix1 : ix2 + 1]).sum())
    if inter <= 0:
        return 0.0
    denom = float(int(area_a) + int(area_b) - inter)
    return float(inter) / denom if denom > 0.0 else 0.0


def _border_touch_sides(mask: np.ndarray) -> int:
    h, w = int(mask.shape[0]), int(mask.shape[1])
    if h <= 0 or w <= 0:
        return 0
    sides = 0
    if bool(mask[0, :].any()):
        sides += 1
    if bool(mask[h - 1, :].any()):
        sides += 1
    if bool(mask[:, 0].any()):
        sides += 1
    if bool(mask[:, w - 1].any()):
        sides += 1
    return sides


def _postprocess_masks(
    masks: List[np.ndarray], scores: List[float], cfg: PostprocessConfig
) -> Tuple[List[np.ndarray], List[float]]:
    filtered: List[Tuple[np.ndarray, float, int, Tuple[int, int, int, int]]] = []

    # Drop overly-large masks (often "background-like" regions in everything mode).
    # This improves instance completeness when building a mutually exclusive label_map.
    img_area: Optional[int] = None

    for m, s in zip(masks, scores):
        area = int(m.sum())
        if img_area is None:
            img_area = int(m.shape[0] * m.shape[1])
        if area < cfg.area_min:
            continue
        if cfg.area_max_ratio > 0 and img_area is not None:
            if area > int(cfg.area_max_ratio * img_area):
                continue

        if cfg.morph_kernel and cfg.morph_kernel > 0:
            k = int(cfg.morph_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            m_u8 = (m.astype(np.uint8) * 255)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, kernel)
            m = m_u8 > 0
            area = int(m.sum())
            if area < cfg.area_min:
                continue

        if cfg.cc_min_area and cfg.cc_min_area > 0:
            m = _filter_mask_connected_components(m, int(cfg.cc_min_area))
            area = int(m.sum())
            if area < cfg.area_min:
                continue

        if cfg.border_touch_min_sides and cfg.border_touch_min_sides > 0:
            if _border_touch_sides(m) >= int(cfg.border_touch_min_sides):
                continue

        bbox = _mask_bbox_xyxy(m)
        if bbox is None:
            continue

        filtered.append((m, float(s), area, bbox))

    if not filtered:
        return [], []

    filtered.sort(key=lambda x: (x[1], x[2]), reverse=True)
    if cfg.max_masks and cfg.max_masks > 0:
        filtered = filtered[: max(cfg.max_masks * 2, cfg.max_masks)]

    selected: List[Tuple[np.ndarray, Tuple[int, int, int, int], float, int]] = []
    for m, s, area, bbox in filtered:
        keep = True
        for sm, sb, ss, sarea in selected:
            if _bbox_iou(bbox, sb) < 0.01:
                continue
            if _mask_iou_from_bbox(m, area, bbox, sm, sarea, sb) > cfg.iou_nms:
                keep = False
                break
        if keep:
            selected.append((m, bbox, float(s), int(area)))
            if cfg.max_masks and len(selected) >= cfg.max_masks:
                break

    selected_masks: List[np.ndarray] = []
    selected_scores: List[float] = []
    for sm, sb, ss, sarea in selected:
        selected_masks.append(sm)
        selected_scores.append(float(ss))

    return selected_masks, selected_scores


def _build_label_map(
    masks: List[np.ndarray],
    scores: Optional[List[float]],
    h: int,
    w: int,
    min_assign_pixels: int = 0,
    min_assign_ratio: float = 0.0,
) -> np.ndarray:
    label = np.full((h, w), -1, dtype=np.int16)
    if not masks:
        return label

    # Overlap resolution contract:
    # - higher score wins (if available)
    # - if score is unavailable, use larger area first
    if scores is None or len(scores) != len(masks):
        scores = [1.0] * len(masks)
    order = sorted(
        range(len(masks)),
        key=lambda i: (-float(scores[i]), -int(masks[i].sum())),
    )
    inst_id = 0
    for i in order:
        m = masks[i]
        if m.dtype != np.bool_:
            m = m.astype(np.bool_)
        mask_area = int(m.sum())
        if mask_area <= 0:
            continue
        free = (label < 0) & m
        if not np.any(free):
            continue
        free_area = int(free.sum())
        if min_assign_pixels and min_assign_pixels > 0:
            if free_area < int(min_assign_pixels):
                continue
        if min_assign_ratio and min_assign_ratio > 0:
            if (float(free_area) / float(mask_area)) < float(min_assign_ratio):
                continue
        label[free] = inst_id
        inst_id += 1

    return label


def _cleanup_label_map_islands(label_map: np.ndarray, island_min_area: int) -> np.ndarray:
    if island_min_area <= 0:
        return label_map
    if label_map.dtype != np.int16:
        label_map = label_map.astype(np.int16, copy=False)

    label = label_map.copy()
    max_id = int(label.max())
    if max_id < 0:
        return label

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for lid in range(max_id + 1):
        m_u8 = (label == lid).astype(np.uint8)
        if int(m_u8.sum()) == 0:
            continue
        num, cc, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=8)
        if num <= 1:
            continue
        for cid in range(1, num):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area >= int(island_min_area):
                continue
            comp = (cc == cid)
            dil = cv2.dilate(comp.astype(np.uint8), kernel, iterations=1).astype(np.bool_)
            ring = np.logical_and(dil, np.logical_not(comp))
            neigh = label[ring]
            if neigh.size == 0:
                label[comp] = -1
                continue
            uniq, counts = np.unique(neigh, return_counts=True)
            # Exclude current label id itself.
            valid = uniq != lid
            uniq = uniq[valid]
            counts = counts[valid]
            if uniq.size == 0:
                label[comp] = -1
                continue
            tgt = int(uniq[int(np.argmax(counts))])
            label[comp] = tgt

    return label


def _looks_like_cuda_failure(e: Exception) -> bool:
    msg = f"{type(e).__name__}: {e}".lower()
    return (
        ("cuda" in msg)
        or ("cudnn" in msg)
        or ("cublas" in msg)
        or ("out of memory" in msg)
        or ("device-side assert" in msg)
    )


class FastSAMServer:
    def __init__(
        self,
        weights: str,
        device: str,
        imgsz: int,
        conf: float,
        iou: float,
        retina_masks: bool,
        post_cfg: PostprocessConfig,
        half: bool = True,
        save_everything_vis: bool = True,
        save_post_vis: bool = True,
    ) -> None:
        self.weights = weights
        self.device = device
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.retina_masks = bool(retina_masks)
        self.post_cfg = post_cfg
        self.half = bool(half)
        self.predictor: Optional[FastSAMPredictor] = None
        self.last_profile: Dict[str, float] = {}

        self.everything_vis_dir: Optional[str] = _default_everything_vis_dir() if bool(save_everything_vis) else None
        if self.everything_vis_dir is not None:
            try:
                os.makedirs(self.everything_vis_dir, exist_ok=True)
            except Exception as e:
                print(
                    f"[fastsam_server] failed to create everything_vis_dir={self.everything_vis_dir}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                self.everything_vis_dir = None

        self.post_vis_dir: Optional[str] = _default_post_processing_vis_dir() if bool(save_post_vis) else None
        if self.post_vis_dir is not None:
            try:
                os.makedirs(self.post_vis_dir, exist_ok=True)
            except Exception as e:
                print(
                    f"[fastsam_server] failed to create post_vis_dir={self.post_vis_dir}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                self.post_vis_dir = None

        with contextlib.redirect_stdout(io.StringIO()):
            self.model = FastSAM(self.weights)
        self._setup_predictor()

    def _use_half(self) -> bool:
        return bool(self.half and self.device != "cpu" and str(self.device).startswith("cuda"))

    def _setup_predictor(self) -> None:
        overrides = self.model.overrides.copy()
        overrides.update(
            {
                "conf": self.conf,
                "iou": self.iou,
                "mode": "predict",
                "task": "segment",
                "save": False,
                "imgsz": self.imgsz,
                "device": self.device,
                "retina_masks": self.retina_masks,
                "verbose": False,
                "half": self._use_half(),
            }
        )
        self.predictor = FastSAMPredictor(overrides=overrides)
        self.predictor.setup_model(model=self.model.model, verbose=False)
        if torch is not None:
            try:
                actual_device = getattr(self.predictor, "device", None)
                cuda_name = ""
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    cuda_name = f" cuda_name={torch.cuda.get_device_name(0)}"
                print(
                    f"[fastsam_server] ready device={actual_device} requested={self.device} "
                    f"half={1 if self._use_half() else 0} imgsz={self.imgsz} retina={1 if self.retina_masks else 0}"
                    f"{cuda_name}",
                    file=sys.stderr,
                )
            except Exception:
                pass

    def _maybe_save_everything_raw_vis(
        self,
        frame_id: int,
        image_rgb: np.ndarray,
        masks_raw: List[np.ndarray],
    ) -> None:
        if self.everything_vis_dir is None:
            return
        if frame_id < 0:
            return
        try:
            vis_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            vis = _render_everything_raw_vis(vis_bgr, masks_raw, alpha=0.45)
            out_path = os.path.join(self.everything_vis_dir, f"frame_{frame_id:06d}.png")
            # Keep a valid image extension for OpenCV codec selection.
            tmp_path = out_path + ".tmp.png"
            ok = cv2.imwrite(tmp_path, vis)
            if not ok:
                raise RuntimeError("imwrite_failed")
            os.replace(tmp_path, out_path)
        except Exception as e:
            print(
                f"[fastsam_server] save_everything_raw_vis failed frame_id={frame_id}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    def _maybe_save_post_processing_vis(
        self,
        frame_id: int,
        image_rgb: np.ndarray,
        label_map: np.ndarray,
    ) -> None:
        if self.post_vis_dir is None:
            return
        if frame_id < 0:
            return
        try:
            vis_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            vis = _render_post_processing_vis(vis_bgr, label_map, alpha=0.35, contour_thickness=1)
            out_path = os.path.join(self.post_vis_dir, f"frame_{frame_id:06d}.png")
            tmp_path = out_path + ".tmp.png"
            ok = cv2.imwrite(tmp_path, vis)
            if not ok:
                raise RuntimeError("imwrite_failed")
            os.replace(tmp_path, out_path)
        except Exception as e:
            print(
                f"[fastsam_server] save_post_vis failed frame_id={frame_id}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )

    def segment_everything(self, image: np.ndarray, is_rgb: bool, frame_id: int = -1) -> np.ndarray:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("invalid image")

        t_start = time.perf_counter()
        h, w = int(image.shape[0]), int(image.shape[1])
        if is_rgb:
            image_rgb = image
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        def _infer() -> Any:
            if self.predictor is None:
                self._setup_predictor()
            assert self.predictor is not None
            return self.predictor(image_bgr, stream=False)

        with contextlib.redirect_stdout(io.StringIO()):
            try:
                t0 = time.perf_counter()
                results = _infer()
                t_infer = time.perf_counter()
                if results and len(results) > 0 and getattr(results[0], "masks", None) is not None:
                    ann = results[0].masks.data
                else:
                    ann = []
            except Exception as e:
                # Best-effort fallback: if CUDA fails, switch to CPU to keep the system running.
                # This matches the robustness requirement in `docs/参考资料/DIF_SLAM_工程实现方案.md` (Module A).
                if self.device != "cpu" and _looks_like_cuda_failure(e):
                    old_device = self.device
                    self.device = "cpu"
                    self.half = False
                    if torch is not None and hasattr(torch, "cuda"):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    print(
                        f"[fastsam_server] CUDA failure on device={old_device}, fallback to cpu: {type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    self._setup_predictor()
                    t0 = time.perf_counter()
                    results = _infer()
                    t_infer = time.perf_counter()
                    if results and len(results) > 0 and getattr(results[0], "masks", None) is not None:
                        ann = results[0].masks.data
                    else:
                        ann = []
                else:
                    raise

        t0_np = time.perf_counter()
        masks_raw, scores_raw = _ann_to_masks_and_scores(ann, h, w)
        t_np = time.perf_counter()
        self._maybe_save_everything_raw_vis(int(frame_id), image_rgb, masks_raw)
        t_raw_vis = time.perf_counter()

        masks, scores = _postprocess_masks(masks_raw, scores_raw, self.post_cfg)
        t_post = time.perf_counter()
        label_map = _build_label_map(
            masks,
            scores,
            h,
            w,
            min_assign_pixels=int(self.post_cfg.min_assign_pixels),
            min_assign_ratio=float(self.post_cfg.min_assign_ratio),
        )
        t_label = time.perf_counter()
        label_map = _cleanup_label_map_islands(label_map, int(self.post_cfg.island_min_area))
        t_cleanup = time.perf_counter()
        self._maybe_save_post_processing_vis(int(frame_id), image_rgb, label_map)
        t_end = time.perf_counter()
        self.last_profile = {
            "infer": (t_infer - t0) * 1000.0,
            "to_numpy": (t_np - t0_np) * 1000.0,
            "raw_vis": (t_raw_vis - t_np) * 1000.0,
            "post_masks": (t_post - t_raw_vis) * 1000.0,
            "label": (t_label - t_post) * 1000.0,
            "cleanup": (t_cleanup - t_label) * 1000.0,
            "post_vis": (t_end - t_cleanup) * 1000.0,
            "total": (t_end - t_start) * 1000.0,
        }
        return label_map


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.9)
    parser.add_argument("--retina-masks", action="store_true")
    parser.add_argument("--half", type=int, default=1)
    parser.add_argument("--deterministic", type=int, default=0)
    parser.add_argument("--save-everything-vis", type=int, default=1)
    parser.add_argument("--save-post-vis", type=int, default=1)
    parser.add_argument("--area-min", type=int, default=600)
    parser.add_argument("--area-max-ratio", type=float, default=0.65)
    parser.add_argument("--border-touch-min-sides", type=int, default=3)
    parser.add_argument("--morph-kernel", type=int, default=3)
    parser.add_argument("--iou-nms", type=float, default=0.9)
    parser.add_argument("--max-masks", type=int, default=50)
    parser.add_argument("--cc-min-area", type=int, default=0)
    parser.add_argument("--min-assign-pixels", type=int, default=0)
    parser.add_argument("--min-assign-ratio", type=float, default=0.0)
    parser.add_argument("--island-min-area", type=int, default=0)
    args = parser.parse_args()
    _configure_torch_runtime(deterministic=(int(args.deterministic) != 0), seed=0)

    post_cfg = PostprocessConfig(
        area_min=args.area_min,
        area_max_ratio=float(args.area_max_ratio),
        border_touch_min_sides=int(args.border_touch_min_sides),
        morph_kernel=args.morph_kernel,
        iou_nms=args.iou_nms,
        max_masks=args.max_masks,
        cc_min_area=int(args.cc_min_area),
        min_assign_pixels=int(args.min_assign_pixels),
        min_assign_ratio=float(args.min_assign_ratio),
        island_min_area=int(args.island_min_area),
    )
    server = FastSAMServer(
        weights=args.weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        retina_masks=bool(args.retina_masks),
        post_cfg=post_cfg,
        half=(int(args.half) != 0),
        save_everything_vis=(int(args.save_everything_vis) != 0),
        save_post_vis=(int(args.save_post_vis) != 0),
    )

    sin = sys.stdin.buffer
    sout = sys.stdout.buffer

    while True:
        try:
            header, payload = _read_message(sin)
        except EOFError:
            return 0
        except Exception as e:
            try:
                _write_message(
                    sout,
                    {"type": "error", "error": f"read_message_failed: {type(e).__name__}: {e}"},
                    b"",
                )
            except BrokenPipeError:
                return 0
            continue

        msg_type = header.get("type", "segment")
        if msg_type == "shutdown":
            try:
                _write_message(sout, {"type": "shutdown_ack"}, b"")
            except BrokenPipeError:
                pass
            return 0
        if msg_type != "segment":
            try:
                _write_message(sout, {"type": "error", "error": f"unknown_type: {msg_type}"}, b"")
            except BrokenPipeError:
                return 0
            continue

        frame_id = int(header.get("frame_id", -1))
        timestamp = float(header.get("timestamp", -1.0))
        is_rgb = bool(int(header.get("is_rgb", 0)))

        t0 = time.time()
        try:
            jpg = np.frombuffer(payload, dtype=np.uint8)
            image = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
            label_map = server.segment_everything(image, is_rgb=is_rgb, frame_id=frame_id)

            label_u16 = (label_map.astype(np.int32) + 1).astype(np.uint16)
            ok, png = cv2.imencode(".png", label_u16)
            if not ok:
                raise RuntimeError("png_encode_failed")

            elapsed_ms = (time.time() - t0) * 1000.0
            profile = getattr(server, "last_profile", {}) or {}
            try:
                _write_message(
                    sout,
                    {
                        "type": "result",
                        "ok": True,
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "h": int(label_u16.shape[0]),
                        "w": int(label_u16.shape[1]),
                        "elapsed_ms": float(elapsed_ms),
                        "profile_total_ms": float(profile.get("total", elapsed_ms)),
                        "profile_infer_ms": float(profile.get("infer", 0.0)),
                        "profile_post_ms": float(profile.get("post_masks", 0.0)),
                    },
                    png.tobytes(),
                )
            except BrokenPipeError:
                return 0
        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000.0
            try:
                _write_message(
                    sout,
                    {
                        "type": "result",
                        "ok": False,
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "error": f"{type(e).__name__}: {e}",
                        "elapsed_ms": float(elapsed_ms),
                    },
                    b"",
                )
            except BrokenPipeError:
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
