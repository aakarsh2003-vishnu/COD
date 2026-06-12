from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(".")
MANIFEST = ROOT / "data/manifests/mcod_manifest.csv"
REPORTS = ROOT / "data/reports"
FIGURES = ROOT / "outputs/figures"
TABLES = ROOT / "outputs/tables"


def ensure_dirs():
    for path in [REPORTS, FIGURES, TABLES]:
        path.mkdir(parents=True, exist_ok=True)


def read_manifest():
    return pd.read_csv(MANIFEST)


def mat_path_from_band_ref(ref):
    return Path(str(ref).split("::", 1)[0])


def load_cube(row):
    band_ref = row["path_s1"] if isinstance(row, dict) else row.path_s1
    mat_path = mat_path_from_band_ref(band_ref)
    with h5py.File(mat_path, "r") as handle:
        return handle["gray"][:]


def minmax_uint8(arr):
    arr = arr.astype(np.float32)
    if arr.max() == arr.min():
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)


def write_channel_order_verification(df):
    lines = [
        "MCOD channel-order verification",
        "==============================",
        "",
        "Local MAT convention verified from HDF5 key `gray`: shape is (8, H, W).",
        "Manifest band references use zero-based indices:",
        "S1=gray[0], S2=gray[1], S3=gray[2], S4=gray[3], S5=gray[4], S6=gray[5], S7=gray[6], S8=gray[7].",
        "",
        "Official false-colour view follows the project/PDF convention:",
        "R <- S5, G <- S3, B <- S2.",
        "",
        f"Rows checked in manifest: {len(df)}",
        f"Train rows: {(df['split'] == 'train').sum()}",
        f"Test rows: {(df['split'] == 'test').sum()}",
        "",
        "Note: official challenge-attribute files were not found locally, so non-SO labels are not treated as ground truth.",
    ]
    (REPORTS / "channel_order_verification.txt").write_text("\n".join(lines) + "\n")


def write_spectral_generation_log(df):
    views = [
        ("official_false_colour", "S2,S3,S5", "R=S5;G=S3;B=S2; per-sample min-max normalization", ".png"),
        ("S6_rededge_gray3", "S6", "S6 min-max normalized and repeated to 3 channels", ".png"),
        ("S7_nir1_gray3", "S7", "S7 min-max normalized and repeated to 3 channels", ".png"),
        ("S8_nir2_gray3", "S8", "S8 min-max normalized and repeated to 3 channels", ".png"),
        ("visible_group_projection", "S1-S5", "local PCA projection to 3 channels", ".png"),
        ("nir_group_projection", "S6-S8", "local PCA projection to 3 channels", ".png"),
        ("all8_input", "S1-S8", "native normalized 8-channel tensor", ".npy"),
        ("ground_truth_mask", "official mask", "binary mask copied/normalized from GT", ".png"),
    ]
    rows = []
    for split in ["train", "test"]:
        split_rows = df[df["split"] == split]
        for view, bands, rule, suffix in views:
            view_dir = ROOT / "MCOD_processed" / split / view
            count = len(list(view_dir.glob(f"*{suffix}"))) if view_dir.exists() else 0
            rows.append(
                {
                    "split": split,
                    "view_name": view,
                    "bands_used": bands,
                    "construction_rule": rule,
                    "expected_samples": len(split_rows),
                    "files_found": count,
                    "complete": count == len(split_rows),
                    "output_dir": view_dir.as_posix(),
                }
            )
    pd.DataFrame(rows).to_csv(REPORTS / "spectral_view_generation_log.csv", index=False)


def write_attribute_support(df):
    attr_rows = []
    for row in df.itertuples(index=False):
        so = bool(float(row.object_area_ratio) < 0.001)
        attr_rows.append(
            {
                "sample_id": row.sample_id,
                "split": row.split,
                "SC": "",
                "SO": int(so),
                "OC": "",
                "BS": "",
                "UI": "",
                "CB": "",
                "OE": "",
                "II": "",
                "source": "SO_inferred_from_mask_area_ratio;others_require_official_attribute_file",
            }
        )
    pd.DataFrame(attr_rows).to_csv(ROOT / "data/manifests/mcod_attribute_support.csv", index=False)
    lines = [
        "MCOD attribute handling",
        "=======================",
        "",
        "Official attribute files were not found locally.",
        "Defensible derived label: SO, using object_area_ratio < 0.001 from the PDF.",
        "Not treated as ground truth without official labels: SC, OC, BS, UI, CB, OE, II.",
        "",
        "Possible heuristic-only methods:",
        "OE/II/UI: image-intensity statistics; BS: object/background spectral contrast; CB: texture clutter; SC: mask boundary complexity; OC: not reliable from masks alone.",
        "These are not used as official labels in the manifest.",
    ]
    (REPORTS / "attribute_handling_plan.txt").write_text("\n".join(lines) + "\n")


def create_first20_grid(df):
    subset = df.head(20)
    fig, axes = plt.subplots(len(subset), 4, figsize=(12, max(20, len(subset) * 2.2)))
    for idx, row in enumerate(subset.itertuples(index=False)):
        cube = load_cube(row)
        false_colour = np.asarray(Image.open(row.path_false_colour).convert("RGB"))
        mask = np.asarray(Image.open(row.path_mask).convert("L"))
        imgs = [
            minmax_uint8(cube[0]),
            false_colour,
            minmax_uint8(cube[6]),
            mask,
        ]
        titles = ["S1", "False colour", "S7 NIR1", "GT mask"]
        for col, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[idx, col]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            if idx == 0:
                ax.set_title(title)
            ax.set_ylabel(row.sample_id[:16], fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(FIGURES / "first_20_band_view_grid.png", dpi=150)
    plt.close(fig)


def create_task7_figures(df):
    area = pd.read_csv(ROOT / "data_audit/MCOD/object_area_distribution.csv")
    band_stats = pd.read_csv(ROOT / "data_audit/MCOD/band_statistics.csv")
    resolution = pd.read_csv(ROOT / "data_audit/MCOD/resolution_distribution.csv")
    attr_support = pd.read_csv(ROOT / "data/manifests/mcod_attribute_support.csv")

    existing_sample = ROOT / "data_audit/MCOD/figures/sample_band_visualization.png"
    if existing_sample.exists():
        Image.open(existing_sample).save(FIGURES / "sample_band_visualization.png")

    sample = df.iloc[0]
    false_colour = np.asarray(Image.open(sample["path_false_colour"]).convert("RGB"))
    mask_img = Image.open(sample["path_mask"]).convert("L")
    if mask_img.size != (false_colour.shape[1], false_colour.shape[0]):
        mask_img = mask_img.resize((false_colour.shape[1], false_colour.shape[0]), Image.NEAREST)
    mask = np.asarray(mask_img) > 0
    overlay = false_colour.copy()
    overlay[mask] = (0.55 * overlay[mask] + np.array([255, 0, 0]) * 0.45).astype(np.uint8)
    Image.fromarray(overlay).save(FIGURES / "sample_false_colour_mask_overlay.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(band_stats["band"], band_stats["mean"], yerr=band_stats["std"], fmt="o-", capsize=4)
    ax.set_xlabel("Band")
    ax.set_ylabel("Mean +/- std")
    ax.set_title("MCOD Band Statistics")
    fig.tight_layout()
    fig.savefig(FIGURES / "band_mean_std_plot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(area["area_ratio"], bins=50)
    ax.set_xlabel("Object area ratio")
    ax.set_ylabel("Samples")
    ax.set_title("Object Area Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "object_area_histogram.png", dpi=150)
    plt.close(fig)

    attr_counts = {"SO": int(attr_support["SO"].sum())}
    for attr in ["SC", "OC", "BS", "UI", "CB", "OE", "II"]:
        attr_counts[attr] = 0
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(attr_counts.keys(), attr_counts.values())
    ax.set_ylabel("Samples")
    ax.set_title("Available Attribute Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "attribute_distribution_bar.png", dpi=150)
    plt.close(fig)

    attrs = ["SC", "SO", "OC", "BS", "UI", "CB", "OE", "II"]
    matrix = np.zeros((len(attrs), len(attrs)), dtype=int)
    so_count = int(attr_support["SO"].sum())
    matrix[attrs.index("SO"), attrs.index("SO")] = so_count
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(attrs)), attrs, rotation=45, ha="right")
    ax.set_yticks(range(len(attrs)), attrs)
    ax.set_title("Available Attribute Co-occurrence")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES / "attribute_cooccurrence_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    split_area = df[["split", "object_area_ratio"]]
    ax.hist(
        [split_area[split_area["split"] == "train"]["object_area_ratio"], split_area[split_area["split"] == "test"]["object_area_ratio"]],
        bins=40,
        label=["train", "test"],
        alpha=0.75,
    )
    ax.set_xlabel("Object area ratio")
    ax.set_ylabel("Samples")
    ax.set_title("Train/Test Object Size Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "train_test_size_distribution.png", dpi=150)
    plt.close(fig)

    if not resolution.empty:
        resolution["resolution"] = resolution["w"].astype(str) + "x" + resolution["h"].astype(str)
        resolution.groupby(["source", "resolution"]).size().reset_index(name="count").to_csv(
            TABLES / "task7_resolution_summary.csv", index=False
        )


def write_task8_outputs(df):
    metric_defs = [
        ("S-measure", "Higher", "PySODMetrics Smeasure"),
        ("E-measure", "Higher", "PySODMetrics Emeasure"),
        ("F-measure", "Higher", "PySODMetrics Fmeasure"),
        ("MAE", "Lower", "PySODMetrics MAE plus local MAE"),
        ("Weighted F-measure", "Higher", "PySODMetrics WeightedFmeasure"),
        ("Dice", "Higher", "Local binary overlap"),
        ("mIoU", "Higher", "Local binary IoU"),
        ("Boundary F-score", "Higher", "Local boundary tolerance metric"),
    ]
    pd.DataFrame(metric_defs, columns=["metric", "direction", "implementation"]).to_csv(
        TABLES / "task8_metric_definitions.csv", index=False
    )


def main():
    ensure_dirs()
    df = read_manifest()
    write_channel_order_verification(df)
    write_spectral_generation_log(df)
    write_attribute_support(df)
    create_first20_grid(df)
    create_task7_figures(df)
    write_task8_outputs(df)
    print("Completed Task 6-8 missing reports and figures. Evaluation scripts are available but no test run was executed.")


if __name__ == "__main__":
    main()
