import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from py_sod_metrics import Emeasure, Fmeasure, MAE, Smeasure, WeightedFmeasure
except ImportError:
    Emeasure = Fmeasure = MAE = Smeasure = WeightedFmeasure = None


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def boundary_fscore(pred, gt):
    def find_boundaries(mask):
        boundary = np.zeros_like(mask, dtype=bool)
        boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
        boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
        boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
        boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        return boundary & mask

    def binary_dilation(mask, radius):
        padded = np.pad(mask, radius, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        size = radius * 2 + 1
        for y in range(size):
            for x in range(size):
                out |= padded[y:y + mask.shape[0], x:x + mask.shape[1]]
        return out

    pred_b = find_boundaries(pred)
    gt_b = find_boundaries(gt)
    if pred_b.sum() == 0 and gt_b.sum() == 0:
        return 1.0
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    radius = max(1, int(round(0.002 * math.hypot(*gt.shape))))
    precision = (pred_b & binary_dilation(gt_b, radius)).sum() / max(pred_b.sum(), 1)
    recall = (gt_b & binary_dilation(pred_b, radius)).sum() / max(gt_b.sum(), 1)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def binary_metrics(pred, gt):
    pred_b = pred > 127
    gt_b = gt > 127
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, ~gt_b).sum()
    fn = np.logical_and(~pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return {
        "dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "miou": float(tp / max(union, 1)),
        "boundary_f": boundary_fscore(pred_b, gt_b),
    }


def local_mae(pred, gt):
    pred = pred.astype(np.float32) / 255.0
    gt = (gt > 127).astype(np.float32)
    return float(np.mean(np.abs(pred - gt)))


def summarize(value):
    if isinstance(value, dict):
        return json.dumps({key: summarize(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        return float(np.nanmean(value.astype(np.float64))) if value.size else math.nan
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float64)
        return float(np.nanmean(arr)) if arr.size else math.nan
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def find_by_stem(root, stem):
    for suffix in IMAGE_EXTENSIONS:
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def rows_from_manifest(manifest_path, split):
    with open(manifest_path, newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    return [(row["sample_id"], Path(row["path_mask"])) for row in rows]


def rows_from_mask_dir(mask_dir):
    rows = []
    for path in sorted(mask_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append((path.stem, path))
    return rows


def evaluate(manifest_path, prediction_dir, output_path, split="test", limit=None, mask_dir=None):
    if mask_dir is not None:
        rows = rows_from_mask_dir(mask_dir)
    elif manifest_path is not None and manifest_path.exists():
        rows = rows_from_manifest(manifest_path, split)
    else:
        raise FileNotFoundError("Provide --mask-dir or an existing --manifest.")

    if not rows:
        raise ValueError("No evaluation rows found. Check --split, --manifest, or --mask-dir.")

    metrics = {}
    if Smeasure is not None:
        metrics = {
            "smeasure": Smeasure(),
            "emeasure": Emeasure(),
            "fmeasure": Fmeasure(),
            "mae": MAE(),
            "weighted_fmeasure": WeightedFmeasure(),
        }
    binary_rows = []
    mae_rows = []
    evaluated = 0
    missing = 0

    for sample_id, mask_path in rows:
        pred_path = find_by_stem(prediction_dir, sample_id)
        if pred_path is None:
            missing += 1
            continue

        pred = np.asarray(Image.open(pred_path).convert("L"))
        gt = np.asarray(Image.open(mask_path).convert("L"))
        gt = np.where(gt > 0, 255, 0).astype(np.uint8)
        if pred.shape != gt.shape:
            pred = np.asarray(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.BILINEAR))

        for metric in metrics.values():
            metric.step(pred=pred, gt=gt, normalize=True)
        binary_rows.append(binary_metrics(pred, gt))
        mae_rows.append(local_mae(pred, gt))
        evaluated += 1

        if limit is not None and evaluated >= limit:
            break

    result = {
        "prediction_dir": prediction_dir.as_posix(),
        "split": split,
        "evaluated_samples": evaluated,
        "missing_predictions": missing,
        "local_mae": float(np.mean(mae_rows)) if mae_rows else math.nan,
    }
    if binary_rows:
        for column in binary_rows[0]:
            result[column] = float(np.mean([row[column] for row in binary_rows]))
    for name, metric in metrics.items():
        result[name] = summarize(metric.get_results())
    if not metrics:
        result["notes"] = "py_sod_metrics is not installed; reported local_mae, dice, miou, and boundary_f only."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate COD prediction maps against MCOD masks.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--mask-dir", default=None)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    result = evaluate(
        manifest_path=Path(args.manifest) if args.manifest else None,
        prediction_dir=Path(args.prediction_dir),
        output_path=Path(args.output),
        split=args.split,
        limit=args.limit,
        mask_dir=Path(args.mask_dir) if args.mask_dir else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
