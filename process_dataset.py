import numpy as np
import cv2
from sklearn.decomposition import PCA
import h5py
import os
from pathlib import Path
from tqdm import tqdm
import json


# ============================================================
# Utility Functions
# ============================================================

def minmax_normalize(band):
    """
    Normalize a single spectral band to uint8 [0,255]
    """
    band = band.astype(np.float32)

    if band.max() == band.min():
        return np.zeros_like(band, dtype=np.uint8)

    band = (band - band.min()) / (band.max() - band.min())
    band = (band * 255).astype(np.uint8)

    return band


def grayscale_to_rgb(band):
    """
    Convert single-band image into H×W×3
    """
    band = minmax_normalize(band)
    return np.stack([band, band, band], axis=-1)


def create_false_colour(s2, s3, s5):
    """
    official_false_colour

    Verify channel order with dataset paper.
    Current convention:
        R <- S5
        G <- S3
        B <- S2
    """
    r = minmax_normalize(s5)
    g = minmax_normalize(s3)
    b = minmax_normalize(s2)

    return np.stack([r, g, b], axis=-1)


def pca_projection(bands, n_components=3):
    """
    PCA projection from N bands to RGB

    Input:
        bands -> list of H×W arrays
    """

    h, w = bands[0].shape

    cube = np.stack(bands, axis=-1).astype(np.float32)

    pixels = cube.reshape(-1, cube.shape[-1])

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(pixels)

    projected = projected.reshape(h, w, n_components)

    rgb = np.zeros_like(projected, dtype=np.uint8)

    for i in range(n_components):
        rgb[..., i] = minmax_normalize(projected[..., i])

    return rgb, pca


def prepare_all8_tensor(bands):
    """
    Native 8-channel tensor

    Output:
        H×W×8 float32 normalized
    """

    cube = np.stack(bands, axis=-1).astype(np.float32)

    for i in range(cube.shape[-1]):
        band = cube[..., i]

        if band.max() > band.min():
            cube[..., i] = (band - band.min()) / (
                band.max() - band.min()
            )

    return cube

def process_mask(mask):
    """
    Keep the official mask unchanged.
    """
    return mask


def align_to_mask(view, mask):
    """
    Align generated spectral views to the official mask orientation.
    MCOD .mat bands are stored transposed relative to GT/Pcolor images.
    """
    if mask is None:
        return view

    mask_shape = mask.shape[:2]
    view_shape = view.shape[:2]

    if view_shape == mask_shape:
        return view

    if view_shape == mask_shape[::-1]:
        if view.ndim == 2:
            return view.T
        if view.ndim == 3:
            return np.transpose(view, (1, 0, 2))

    raise ValueError(
        f"Cannot align view shape {view_shape} to mask shape {mask_shape}"
    )


# ============================================================
# Main View Generator
# ============================================================

def generate_views(
    S1, S2, S3, S4, S5, S6, S7, S8,
    mask=None
):

    views = {}

    # --------------------------------------------------------
    # official_false_colour
    # --------------------------------------------------------
    views["official_false_colour"] = create_false_colour(
        S2, S3, S5
    )

    # --------------------------------------------------------
    # Single-band grayscale views
    # --------------------------------------------------------
    views["S6_rededge_gray3"] = grayscale_to_rgb(S6)

    views["S7_nir1_gray3"] = grayscale_to_rgb(S7)

    views["S8_nir2_gray3"] = grayscale_to_rgb(S8)

    # --------------------------------------------------------
    # Visible group projection
    # S1-S5
    # --------------------------------------------------------
    visible_rgb, visible_pca = pca_projection(
        [S1, S2, S3, S4, S5]
    )

    views["visible_group_projection"] = visible_rgb

    # --------------------------------------------------------
    # NIR group projection
    # S6-S8
    # --------------------------------------------------------
    nir_rgb, nir_pca = pca_projection(
        [S6, S7, S8]
    )

    views["nir_group_projection"] = nir_rgb

    # --------------------------------------------------------
    # Native 8-band tensor
    # --------------------------------------------------------
    views["all8_input"] = prepare_all8_tensor(
        [S1, S2, S3, S4, S5, S6, S7, S8]
    )

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------
    if mask is not None:
        for view_name in list(views.keys()):
            views[view_name] = align_to_mask(views[view_name], mask)
        views["ground_truth_mask"] = process_mask(mask)

    return views, {
        "visible_pca": visible_pca,
        "nir_pca": nir_pca
    }


# ============================================================
# Data Loading & Processing Pipeline
# ============================================================

def load_bands_from_mat(mat_path):
    """
    Load 8 spectral bands from .mat file (HDF5 format).
    
    Input:
        mat_path -> path to .mat file
    
    Output:
        tuple of (S1, S2, S3, S4, S5, S6, S7, S8) as numpy arrays
    """
    with h5py.File(mat_path, 'r') as f:
        bands_cube = f['gray'][:]  # Shape: (8, H, W)
    
    # Convert from (8, H, W) to individual bands
    S1 = bands_cube[0]
    S2 = bands_cube[1]
    S3 = bands_cube[2]
    S4 = bands_cube[3]
    S5 = bands_cube[4]
    S6 = bands_cube[5]
    S7 = bands_cube[6]
    S8 = bands_cube[7]
    
    return S1, S2, S3, S4, S5, S6, S7, S8


def load_mask_from_image(mask_path):
    """
    Load ground truth mask from image file.
    
    Input:
        mask_path -> path to mask image (jpg, png, etc.)
    
    Output:
        mask as numpy array
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask


def process_single_sample(
    sample_id,
    mat_path,
    mask_path,
    output_base_dir,
    split_name="train"
):
    """
    Process a single multispectral sample:
    - Load 8-band data
    - Generate all views
    - Save all8_input as NPY binary file
    - Save RGB views as PNG images
    
    Input:
        sample_id -> identifier for the sample
        mat_path -> path to .mat file with 8 bands
        mask_path -> path to ground truth mask
        output_base_dir -> base directory for MCOD_processed
        split_name -> "train" or "test"
    
    Output:
        dict with processing status
    """
    try:
        # Load data
        S1, S2, S3, S4, S5, S6, S7, S8 = load_bands_from_mat(mat_path)
        mask = load_mask_from_image(mask_path)
        
        # Generate views
        views, pca_dict = generate_views(S1, S2, S3, S4, S5, S6, S7, S8, mask)
        
        # Create output directories
        output_paths = {}
        for view_name in views.keys():
            view_dir = os.path.join(output_base_dir, split_name, view_name)
            Path(view_dir).mkdir(parents=True, exist_ok=True)
            output_paths[view_name] = view_dir
        
        # Save all views (NPY for all8_input, PNG for RGB images)
        saved_files = {}
        for view_name, view_data in views.items():
            if view_name == "all8_input":
                # Save as NPY binary file
                output_file = os.path.join(
                    output_paths[view_name],
                    f"{sample_id}.npy"
                )
                np.save(output_file, view_data)
            else:
                # Save as PNG image file
                output_file = os.path.join(
                    output_paths[view_name],
                    f"{sample_id}.png"
                )
                # Convert to uint8 if not already (for image saving)
                if view_data.dtype != np.uint8:
                    view_data = (view_data * 255).astype(np.uint8) if view_data.max() <= 1 else view_data.astype(np.uint8)
                if view_data.ndim == 2:
                    cv2.imwrite(output_file, view_data)
                else:
                    cv2.imwrite(output_file, cv2.cvtColor(view_data, cv2.COLOR_RGB2BGR))
            saved_files[view_name] = output_file
        
        # Save PCA projection matrices as JSON (metadata)
        metadata = {
            "sample_id": sample_id,
            "split": split_name,
            "views_generated": list(views.keys()),
            "visible_pca_explained_variance": pca_dict["visible_pca"].explained_variance_ratio_.tolist(),
            "nir_pca_explained_variance": pca_dict["nir_pca"].explained_variance_ratio_.tolist(),
        }
        
        metadata_file = os.path.join(output_base_dir, split_name, ".metadata", f"{sample_id}.json")
        Path(metadata_file).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            "status": "success",
            "sample_id": sample_id,
            "split": split_name,
            "files_saved": saved_files
        }
    
    except Exception as e:
        return {
            "status": "error",
            "sample_id": sample_id,
            "split": split_name,
            "error": str(e)
        }


def process_dataset_batch(
    dataset_root="data/MCOD_resized",
    output_root="MCOD_processed",
    splits=["TrainDataset", "TestDataset"]
):
    """
    Process entire MCOD_resized dataset and save to MCOD_processed.
    
    Input:
        dataset_root -> path to MCOD_resized folder
        output_root -> path to MCOD_processed folder
        splits -> list of dataset splits to process (e.g., ["TrainDataset", "TestDataset"])
    
    Output:
        Summary report with processing results
    """
    
    report = {
        "total_samples": 0,
        "successful": 0,
        "failed": 0,
        "errors": [],
        "results_by_split": {}
    }
    
    for split in splits:
        split_lower = split.replace("Dataset", "").lower()
        split_dir = os.path.join(dataset_root, split)
        mat_dir = os.path.join(split_dir, "Mat")
        gt_dir = os.path.join(split_dir, "GT")
        
        if not os.path.exists(mat_dir):
            print(f"⚠️  {split} Mat directory not found: {mat_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {split}...")
        print(f"{'='*60}")
        
        # Get list of .mat files
        mat_files = sorted([f for f in os.listdir(mat_dir) if f.endswith('.mat')])
        
        split_results = {
            "total": len(mat_files),
            "successful": 0,
            "failed": 0,
            "samples": []
        }
        
        # Process each sample
        for mat_file in tqdm(mat_files, desc=f"Processing {split}"):
            sample_id = mat_file.replace('.mat', '')
            mat_path = os.path.join(mat_dir, mat_file)
            
            # Find corresponding mask file
            mask_candidates = [
                os.path.join(gt_dir, f"{sample_id}.jpg"),
                os.path.join(gt_dir, f"{sample_id}.png"),
                os.path.join(gt_dir, f"{sample_id}.JPG"),
            ]
            mask_path = None
            for candidate in mask_candidates:
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            
            if mask_path is None:
                result = {
                    "status": "error",
                    "sample_id": sample_id,
                    "split": split_lower,
                    "error": "Corresponding mask file not found"
                }
                split_results["failed"] += 1
            else:
                result = process_single_sample(
                    sample_id,
                    mat_path,
                    mask_path,
                    output_root,
                    split_name=split_lower
                )
                
                if result["status"] == "success":
                    split_results["successful"] += 1
                else:
                    split_results["failed"] += 1
            
            split_results["samples"].append(result)
        
        report["results_by_split"][split_lower] = split_results
        report["total_samples"] += split_results["total"]
        report["successful"] += split_results["successful"]
        report["failed"] += split_results["failed"]
        
        print(f"\n{split} Summary:")
        print(f"  ✓ Successful: {split_results['successful']}/{split_results['total']}")
        print(f"  ✗ Failed: {split_results['failed']}/{split_results['total']}")
    
    # Save overall report
    report_file = os.path.join(output_root, "processing_report.json")
    Path(report_file).parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples processed: {report['total_samples']}")
    print(f"✓ Successful: {report['successful']}")
    print(f"✗ Failed: {report['failed']}")
    print(f"Report saved to: {report_file}")
    print(f"{'='*60}\n")
    
    return report


if __name__ == "__main__":
    # Run the full processing pipeline
    report = process_dataset_batch(
        dataset_root="data/MCOD_resized",
        output_root="MCOD_processed",
        splits=["TrainDataset", "TestDataset"]
    )
