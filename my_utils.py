import os
import re
import glob
import json
import math
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from bisect import bisect_left
import numpy as np


# =========================================================
# 1) 데이터 준비: 이미지-조인트 매칭 및 인덱싱
# =========================================================

IMG_RE = re.compile(r"zed_(?P<serial>\d+)_(?P<view>[a-zA-Z]+)_(?P<ts>\d+\.\d+)\.jpg$")
JNT_RE = re.compile(r"joint_(?P<robotserial>\d+)_(?P<ts>\d+\.\d+)\.json$")

def parse_img_fname(path: str) -> Optional[Dict[str, Any]]:
    m = IMG_RE.search(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    d["timestamp"] = float(d.pop("ts"))
    d["path"] = path
    return d

def parse_joint_fname(path: str) -> Optional[Dict[str, Any]]:
    m = JNT_RE.search(os.path.basename(path))
    if not m:
        return None
    d = m.groupdict()
    d["timestamp"] = float(d.pop("ts"))
    d["path"] = path
    return d

def flatten_json(prefix: str, obj: Any) -> Dict[str, Any]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_json(f"{prefix}.{k}" if prefix else str(k), v))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten_json(f"{prefix}.{i}" if prefix else str(i), v))
    else:
        out[prefix] = obj
    return out

def load_joint_angles(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return {"joint_load_error": str(e)}
    candidates = ["joint_angles", "joints", "angles", "q", "positions", "joint"]
    for key in candidates:
        if isinstance(data, dict) and key in data:
            return flatten_json("joint", data[key])
    return flatten_json("joint", data)

def build_joint_index(joint_dir: str) -> Tuple[List[float], List[str]]:
    jpaths = sorted(glob.glob(os.path.join(joint_dir, "joint_*.json")))
    ts_list, path_list = [], []
    for p in jpaths:
        info = parse_joint_fname(p)
        if info is None:
            continue
        ts_list.append(info["timestamp"])
        path_list.append(info["path"])
    pairs = sorted(zip(ts_list, path_list), key=lambda x: x[0])
    return [t for t, _ in pairs], [p for _, p in pairs]

def find_nearest_joint_any(ts: float, ts_index: List[float], paths: List[str]):
    if not ts_index:
        return None
    pos = bisect_left(ts_index, ts)
    best = None
    for idx in (pos - 1, pos):
        if 0 <= idx < len(ts_index):
            jt = ts_index[idx]
            dt = abs(jt - ts)
            cand = (jt, paths[idx], dt)
            if (best is None) or (dt < best[2]):
                best = cand
    return best

def scan_images(img_dirs: List[str]) -> List[Dict[str, Any]]:
    imgs = []
    for d in img_dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.jpg"))):
            info = parse_img_fname(p)
            if info:
                imgs.append(info)
    return imgs

def process_dataset_indexing(dataset_root: str, max_time_diff: float = 0.2):
    img_dirs = [os.path.join(dataset_root, x) for x in ("left", "right", "top")]
    joint_dir = os.path.join(dataset_root, "joint")
    out_csv = os.path.join(dataset_root, "matched_index.csv")
    out_jsonl = os.path.join(dataset_root, "matched_index.jsonl")

    joint_ts_index, joint_paths = build_joint_index(joint_dir)
    images = scan_images(img_dirs)

    if not images:
        print(f"[{os.path.basename(dataset_root)}] 이미지가 없습니다.")
        return
    if not joint_ts_index:
        print(f"[{os.path.basename(dataset_root)}] 조인트(JSON)가 없습니다.")
        return

    records = []
    unmatched_too_far = 0
    unmatched_no_joint = 0

    for img in images:
        ts_img = img["timestamp"]
        nearest = find_nearest_joint_any(ts_img, joint_ts_index, joint_paths)

        if nearest is None:
            print(f"[{os.path.basename(dataset_root)}] UNMATCHED(no_joint): {img['path']}")
            unmatched_no_joint += 1
            continue

        joint_ts, joint_path, dt = nearest

        if dt > max_time_diff:
            print(
                f"[{os.path.basename(dataset_root)}] UNMATCHED(threshold) "
                f"dt={dt:.9f}s > {max_time_diff:.9f}s | img_ts={ts_img:.9f} "
                f"-> nearest_joint={os.path.basename(joint_path)} (joint_ts={joint_ts:.9f})"
            )
            unmatched_too_far += 1
            continue

        joint_cols = load_joint_angles(joint_path)
        rec = {
            "img.path": img["path"],
            "img.serial": img["serial"],
            "img.view": img["view"],
            "img.ts": ts_img,
            "joint.path": joint_path,
            "joint.ts": joint_ts,
            "abs_dt": dt
        }
        rec.update(joint_cols)
        records.append(rec)

    if not records:
        print(
            f"[{os.path.basename(dataset_root)}] 매칭된 쌍이 없음 "
            f"(threshold={max_time_diff}s, images={len(images)}, "
            f"too_far={unmatched_too_far}, no_joint={unmatched_no_joint})"
        )
        return

    df = pd.DataFrame(records).sort_values(by=["img.ts"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(
        f"[{os.path.basename(dataset_root)}] 완료: "
        f"matched={len(records)} / images={len(images)} "
        f"(too_far={unmatched_too_far}, no_joint={unmatched_no_joint}) "
        f"-> {out_csv}, {out_jsonl}"
    )
    

# =========================================================
# 2) Dataset helper functions
# =========================================================
def mask_to_bbox(mask: np.ndarray, min_area: int = 100) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    if (x2 - x1 + 1) * (y2 - y1 + 1) < min_area:
        return None
    return x1, y1, x2, y2

def crop_with_padding(img: np.ndarray, box: Tuple[int, int, int, int], pad: int = 10) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(W - 1, x2 + pad)
    y2 = min(H - 1, y2 + pad)
    return img[y1:y2+1, x1:x2+1]

def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,    ca,     d],
        [0.0,  0.0,   0.0,   1.0]
    ], dtype=np.float64)

def angle_to_joint_coordinate(joint_angles_deg: List[float]) -> np.ndarray:
    fr5_dh_parameters = [
        {'alpha':   90, 'a':  0.0, 'd': 0.152, 'theta_offset': 0},
        {'alpha':    0, 'a': -0.425, 'd': 0.0, 'theta_offset': 0},
        {'alpha':    0, 'a': -0.395, 'd': 0.0, 'theta_offset': 0},
        {'alpha':   90, 'a':  0.0, 'd': 0.102, 'theta_offset': 0},
        {'alpha':  -90, 'a':  0.0, 'd': 0.102, 'theta_offset': 0},
        {'alpha':    0, 'a':  0.0, 'd': 0.100, 'theta_offset': 0},
    ]
    assert len(joint_angles_deg) == 6, "joint_angles_deg must have length 6."
    T_cum = np.eye(4, dtype=np.float64)
    joints_xyz = [T_cum[:3, 3].copy()]
    for i in range(len(fr5_dh_parameters)):
        p = fr5_dh_parameters[i]
        alpha = math.radians(p['alpha'])
        theta = math.radians(joint_angles_deg[i] + p['theta_offset'])
        a, d = p['a'], p['d']
        A_i = dh_transform(a, alpha, d, theta)
        T_cum = T_cum @ A_i
        joints_xyz.append(T_cum[:3, 3].copy())
    joints_xyz = np.stack(joints_xyz, axis=0)
    return joints_xyz

def maybe_to_degrees(joint_angles: List[float]) -> List[float]:
    if all(abs(a) <= math.pi * 1.25 for a in joint_angles):
        return [math.degrees(a) for a in joint_angles]
    return joint_angles

def extract_joint_angles_from_row(row: pd.Series) -> Optional[List[float]]:
    keys_idx = [f"joint.{i}" for i in range(6)]
    if all(k in row for k in keys_idx):
        vals = [float(row[k]) for k in keys_idx]
        return maybe_to_degrees(vals)
    keys_q = [f"joint.q{i+1}" for i in range(6)]
    if all(k in row for k in keys_q):
        vals = [float(row[k]) for k in keys_q]
        return maybe_to_degrees(vals)
    joint_cols = [k for k in row.index if isinstance(k, str) and k.startswith("joint.")]
    nums = []
    for k in sorted(joint_cols):
        v = row[k]
        try:
            v = float(v)
            nums.append(v)
        except Exception:
            continue
    if len(nums) >= 6:
        return maybe_to_degrees(nums[:6])
    return None

# =========================================================
# 3) model helper functions
# =========================================================

def reshape_vit_output(x, patch_size=16, img_size=512):
    if x.ndim == 3 and x.shape[1] > 1:
        if x.shape[1] == (img_size // patch_size)**2 + 1:
            x = x[:, 1:]
        B, N, D = x.shape
        H_feat = W_feat = int(N**0.5)
        if H_feat * W_feat != N:
            raise ValueError(f"ViT output sequence length {N} is not a perfect square.")
        x = x.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported ViT output shape for reshaping: {x.shape}")
    return x