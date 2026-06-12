from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SHOW_FIGURES = False

IMG_DIR = ROOT / "MCOD_processed/test/official_false_colour"
GT_DIR = ROOT / "MCOD_processed/test/ground_truth_mask"

SAVE_DIR = ROOT / "visual_outputs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def index_files(folder):
    folder = Path(folder)

    if not folder.exists():
        print(f"Folder does not exist: {folder}")
        return {}

    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def read_gray(path):
    return np.asarray(Image.open(path).convert("L"))


def mask_object_size(mask_path):
    """
    Count foreground/object pixels in the ground-truth mask.
    Assumes object pixels are non-zero.
    """
    mask = read_gray(mask_path)
    return int(np.count_nonzero(mask > 0))


def choose_samples(pred_dir, n=10):
    images = index_files(IMG_DIR)
    masks = index_files(GT_DIR)
    preds = index_files(pred_dir)

    matched_stems = sorted(set(images) & set(masks) & set(preds))

    if not matched_stems:
        return [], images, masks, preds, {}

    size_info = {
        stem: mask_object_size(masks[stem])
        for stem in matched_stems
    }

    sorted_stems = sorted(
        matched_stems,
        key=lambda stem: size_info[stem],
        reverse=True
    )

    return sorted_stems[:n], images, masks, preds, size_info


def safe_name(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace(":", "")
    )


def visualize_prediction_set(title, pred_dir, n=10, save=True, show=True):
    pred_dir = ROOT / pred_dir

    stems, images, masks, preds, size_info = choose_samples(pred_dir, n=n)

    matched_count = len(set(images) & set(masks) & set(preds))

    print("\n" + "=" * 80)
    print(title)
    print(f"Prediction folder: {pred_dir}")
    print(f"Matched samples available: {matched_count}")
    print(f"Showing top {len(stems)} samples sorted by object size: largest to smallest")

    if not stems:
        print("No matched samples found. Check prediction path and filenames.")
        return

    fig, axes = plt.subplots(
        len(stems),
        3,
        figsize=(12, 3.2 * len(stems)),
        constrained_layout=True
    )

    if len(stems) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, stem in enumerate(stems):
        object_pixels = size_info[stem]

        image = read_rgb(images[stem])
        gt_mask = read_gray(masks[stem])
        pred_mask = read_gray(preds[stem])

        panels = [
            ("False-colour image", image, None),
            ("Ground truth", gt_mask, "gray"),
            ("Prediction", pred_mask, "gray"),
        ]

        for col, (label, arr, cmap) in enumerate(panels):
            ax = axes[row, col]

            if arr.ndim == 2:
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=255)
            else:
                ax.imshow(arr)

            ax.set_title(label if row == 0 else "", fontsize=11)

            if col == 0:
                ax.set_ylabel(
                    f"{stem}\nGT pixels: {object_pixels}",
                    fontsize=8
                )
            else:
                ax.set_ylabel("")

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        f"{title}\nTop {len(stems)} largest ground-truth objects",
        fontsize=14,
        fontweight="bold"
    )

    if save:
        save_path = SAVE_DIR / f"{safe_name(title)}_top_{len(stems)}_largest_objects.png"
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved visualization to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ======================================================================
# Visualize all prediction sets
# ======================================================================

visualize_prediction_set(
    title="SINet-V2 fine-tuned false-colour predictions",
    pred_dir="SINet-V2/res/SINet_V2_MCOD_finetune",
    n=10,
    save=True,
    show=SHOW_FIGURES,
)


visualize_prediction_set(
    title="SINet-V2 MSI8 predictions",
    pred_dir="SINet-V2/outputs/predictions/sinetv2_msi8",
    n=10,
    save=True,
    show=SHOW_FIGURES,
)


visualize_prediction_set(
    title="CamoFormer scratch false-colour predictions",
    pred_dir="CamoFormer/output/Prediction/CamoFormer_MCOD_scratch/MCOD_scratch",
    n=10,
    save=True,
    show=SHOW_FIGURES,
)


visualize_prediction_set(
    title="CamoFormer fine-tuned false-colour predictions",
    pred_dir="CamoFormer/output/Prediction/CamoFormer_MCOD_finetune/MCOD_finetune",
    n=10,
    save=True,
    show=SHOW_FIGURES,
)


visualize_prediction_set(
    title="CamoFormer MSI8 predictions",
    pred_dir="CamoFormer/outputs/predictions/camoformer_msi8",
    n=10,
    save=True,
    show=SHOW_FIGURES,
)


visualize_prediction_set(
    title="ZoomNet fine-tuned false-colour predictions",
    pred_dir="ZoomNet/output/ZoomNet_RGB_MCOD_Results/mcod_te",
    n=10,
    save=True,
    show=SHOW_FIGURES,
)


visualize_prediction_set(
    title="ZoomNet MSI8 predictions",
    pred_dir="ZoomNet/output/ZoomNet_MSI8_MCOD_Results/mcod_msi8_te",
    n=10,
    save=True,
    show=SHOW_FIGURES,
)
