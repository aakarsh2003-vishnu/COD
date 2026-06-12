import argparse
import csv
import hashlib
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


REQUIRED_COLUMNS = [
    "sample_id",
    "split",
    "path_s1",
    "path_s2",
    "path_s3",
    "path_s4",
    "path_s5",
    "path_s6",
    "path_s7",
    "path_s8",
    "path_false_colour",
    "path_s6_rededge_gray3",
    "path_s7_nir1_gray3",
    "path_s8_nir2_gray3",
    "path_visible_group_projection",
    "path_nir_group_projection",
    "path_all8_input",
    "path_mask",
    "attributes",
    "height",
    "width",
    "dtype",
    "min_s1",
    "max_s1",
    "min_s8",
    "max_s8",
    "object_area_ratio",
    "duplicate_group",
    "notes",
]


SPLITS = {
    "TrainDataset": "train",
    "TestDataset": "test",
}


PROCESSED_VIEWS = {
    "path_false_colour": ("official_false_colour", ".png"),
    "path_s6_rededge_gray3": ("S6_rededge_gray3", ".png"),
    "path_s7_nir1_gray3": ("S7_nir1_gray3", ".png"),
    "path_s8_nir2_gray3": ("S8_nir2_gray3", ".png"),
    "path_visible_group_projection": ("visible_group_projection", ".png"),
    "path_nir_group_projection": ("nir_group_projection", ".png"),
    "path_all8_input": ("all8_input", ".npy"),
}


def read_gray_cube(mat_path):
    with h5py.File(mat_path, "r") as handle:
        cube = handle["gray"][:]
    if cube.ndim != 3 or cube.shape[0] != 8:
        raise ValueError(f"Expected gray cube with shape (8,H,W), got {cube.shape}")
    return cube


def find_mask(gt_dir, sample_id):
    return find_image(gt_dir, sample_id)


def find_image(image_dir, sample_id):
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF"):
        candidate = image_dir / f"{sample_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def object_area_ratio(mask_path):
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return float(np.count_nonzero(mask > 0) / mask.size)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_duplicate_groups(mat_paths):
    by_hash = {}
    for path in mat_paths:
        by_hash.setdefault(file_hash(path), []).append(path)

    duplicate_groups = {}
    group_id = 1
    for paths in by_hash.values():
        if len(paths) < 2:
            continue
        name = f"exact_dup_{group_id:04d}"
        for path in paths:
            duplicate_groups[path] = name
        group_id += 1

    return duplicate_groups


def existing_processed_path(processed_root, split, view, sample_id, suffix):
    path = processed_root / split / view / f"{sample_id}{suffix}"
    return path if path.exists() else None


def image_size(path):
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def build_rows(dataset_root, processed_root):
    mat_paths = []
    for split_dir_name in SPLITS:
        mat_paths.extend(sorted((dataset_root / split_dir_name / "Mat").glob("*.mat")))
    duplicate_groups = build_duplicate_groups(mat_paths)

    rows = []
    for split_dir_name, split in SPLITS.items():
        split_root = dataset_root / split_dir_name
        mat_dir = split_root / "Mat"
        gt_dir = split_root / "GT"
        pcolor_dir = split_root / "Pcolor"

        for mat_path in sorted(mat_dir.glob("*.mat")):
            sample_id = mat_path.stem
            mask_path = find_mask(gt_dir, sample_id)
            if mask_path is None:
                raise FileNotFoundError(f"Missing GT mask for {sample_id}")
            pcolor_path = find_image(pcolor_dir, sample_id)
            if pcolor_path is None:
                raise FileNotFoundError(f"Missing Pcolor image for {sample_id}")

            cube = read_gray_cube(mat_path)
            _, cube_height, cube_width = cube.shape
            height, width = image_size(mask_path)

            processed_paths = {}
            for column, (view, suffix) in PROCESSED_VIEWS.items():
                processed_path = existing_processed_path(processed_root, split, view, sample_id, suffix)
                processed_paths[column] = processed_path.as_posix() if processed_path else ""

            if not processed_paths["path_false_colour"]:
                processed_paths["path_false_colour"] = pcolor_path.as_posix()
            processed_mask = existing_processed_path(
                processed_root, split, "ground_truth_mask", sample_id, ".png"
            )
            manifest_mask_path = processed_mask if processed_mask else mask_path

            notes = []
            missing_views = [
                view for column, (view, _) in PROCESSED_VIEWS.items()
                if not processed_paths[column]
            ]
            if missing_views:
                notes.append("processed_views_missing:" + "|".join(missing_views))
            if processed_mask is None:
                notes.append("processed_mask_missing_using_raw_gt")
            notes.append("bands_stored_in_mat_gray_cube_indices_0_to_7")
            if (cube_height, cube_width) == (width, height):
                notes.append("mat_gray_cube_transposed_relative_to_mask")
            notes.append("official_attribute_file_not_available_locally")

            band_refs = {
                f"path_s{index + 1}": f"{mat_path.as_posix()}::gray[{index}]"
                for index in range(8)
            }

            row = {
                "sample_id": sample_id,
                "split": split,
                **band_refs,
                **processed_paths,
                "path_mask": manifest_mask_path.as_posix(),
                "attributes": "",
                "height": height,
                "width": width,
                "dtype": str(cube.dtype),
                "min_s1": float(np.min(cube[0])),
                "max_s1": float(np.max(cube[0])),
                "min_s8": float(np.min(cube[7])),
                "max_s8": float(np.max(cube[7])),
                "object_area_ratio": object_area_ratio(mask_path),
                "duplicate_group": duplicate_groups.get(mat_path, ""),
                "notes": ";".join(notes),
            }
            rows.append(row)

    return rows


def write_manifest(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate the MCOD Task 6.3 manifest.")
    parser.add_argument("--dataset-root", default="data/MCOD_resized")
    parser.add_argument("--processed-root", default="MCOD_processed")
    parser.add_argument("--output", default="scripts/data/manifests/mcod_manifest.csv")
    args = parser.parse_args()

    rows = build_rows(Path(args.dataset_root), Path(args.processed_root))
    write_manifest(rows, Path(args.output))
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
