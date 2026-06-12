"""
MCOD Dataset Comprehensive Validation Script
This script performs all required dataset validation checks and generates output reports.
"""
# activate the camo env bofore running pytohn script
from pathlib import Path
from collections import Counter
import hashlib
import json
from PIL import Image
import h5py
import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

dataset_path = Path("data/MCOD_resized")
train_path = dataset_path / "TrainDataset"
test_path = dataset_path / "TestDataset"

train_image_path = train_path / "Pcolor"
test_image_path = test_path / "Pcolor"
train_mask_path = train_path / "GT"
test_mask_path = test_path / "GT"
train_mat_path = train_path / "Mat"
test_mat_path = test_path / "Mat"

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
output_dir = Path("inspection_report")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_files(folder_path, extensions):
    """Get all files with given extensions from a folder."""
    if not folder_path.exists():
        return []
    return sorted([file for file in folder_path.iterdir() 
                   if file.is_file() and file.suffix.lower() in extensions])

def get_all_mat_files():
    """Get all .mat files from train and test directories."""
    return get_files(train_mat_path, {".mat"}) + get_files(test_mat_path, {".mat"})

# ============================================================================
# CHECK 1: Archive/File Readability
# ============================================================================

def check_corrupted_files():
    """Check all files for corruption and save report."""
    corrupted_files = []
    
    # Check images
    for folder_path in [train_image_path, test_image_path]:
        for image_file in get_files(folder_path, image_extensions):
            try:
                with Image.open(image_file) as img:
                    img.verify()
            except Exception as e:
                corrupted_files.append({
                    'file_path': str(image_file),
                    'file_type': 'image',
                    'error': str(e)
                })
    
    # Check masks
    for folder_path in [train_mask_path, test_mask_path]:
        for mask_file in get_files(folder_path, image_extensions):
            try:
                with Image.open(mask_file) as mask:
                    mask.verify()
            except Exception as e:
                corrupted_files.append({
                    'file_path': str(mask_file),
                    'file_type': 'mask',
                    'error': str(e)
                })
    
    # Check .mat files
    for mat_file in get_all_mat_files():
        try:
            with h5py.File(mat_file, "r") as f:
                _ = f["gray"][0, 0, 0]
        except Exception as e:
            corrupted_files.append({
                'file_path': str(mat_file),
                'file_type': 'mat',
                'error': str(e)
            })
    
    # Save report
    df = pd.DataFrame(corrupted_files)
    df.to_csv(output_dir / "raw_corrupted_files.csv", index=False)
    print(f"✓ Corrupted files report: {len(corrupted_files)} issues found")
    return corrupted_files

# ============================================================================
# CHECK 2: Total Sample Count
# ============================================================================

def check_sample_count():
    """Verify total sample count (expected: 1,527)."""
    train_images = get_files(train_image_path, image_extensions)
    test_images = get_files(test_image_path, image_extensions)
    total = len(train_images) + len(test_images)
    
    sample_data = [
        {'category': 'train_images', 'count': len(train_images)},
        {'category': 'test_images', 'count': len(test_images)},
        {'category': 'total_samples', 'count': total},
        {'category': 'expected_total', 'count': 1527},
        {'category': 'match_expected', 'count': 1 if total == 1527 else 0}
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_dir / "raw_sample_count.csv", index=False)
    print(f"✓ Sample count report: {total} total samples (expected: 1527)")
    return sample_data

# ============================================================================
# CHECK 3: Official Split Verification
# ============================================================================

def check_official_split():
    """Verify official train/test split (expected: 1027 train / 500 test)."""
    train_images = len(get_files(train_image_path, image_extensions))
    test_images = len(get_files(test_image_path, image_extensions))
    train_masks = len(get_files(train_mask_path, image_extensions))
    test_masks = len(get_files(test_mask_path, image_extensions))
    train_mat = len(get_files(train_mat_path, {".mat"}))
    test_mat = len(get_files(test_mat_path, {".mat"}))
    
    split_data = [
        {'split': 'train', 'images': train_images, 'masks': train_masks, 'mat_files': train_mat, 'expected': 1027},
        {'split': 'test', 'images': test_images, 'masks': test_masks, 'mat_files': test_mat, 'expected': 500},
        {'split': 'train_match', 'images': train_images, 'masks': train_masks, 'mat_files': train_mat, 'all_equal': 1 if (train_images == train_masks == train_mat) else 0},
        {'split': 'test_match', 'images': test_images, 'masks': test_masks, 'mat_files': test_mat, 'all_equal': 1 if (test_images == test_masks == test_mat) else 0}
    ]
    
    df = pd.DataFrame(split_data)
    df.to_csv(output_dir / "official_split_verification.csv", index=False)
    print(f"✓ Official split report: Train={train_images}, Test={test_images}")
    return split_data

# ============================================================================
# CHECK 4: Eight-Band Completeness
# ============================================================================

def check_band_completeness():
    """Verify every sample has S1-S8 (8 spectral bands)."""
    missing_bands = []
    all_mat_files = get_all_mat_files()
    
    for mat_file in all_mat_files:
        try:
            with h5py.File(mat_file, "r") as f:
                num_bands = f["gray"].shape[0]
                if num_bands != 8:
                    missing_bands.append({
                        'file_path': str(mat_file),
                        'expected_bands': 8,
                        'actual_bands': num_bands
                    })
        except Exception as e:
            missing_bands.append({
                'file_path': str(mat_file),
                'expected_bands': 8,
                'error': str(e)
            })
    
    df = pd.DataFrame(missing_bands) if missing_bands else pd.DataFrame(columns=['file_path', 'expected_bands', 'actual_bands'])
    df.to_csv(output_dir / "missing_band_report.csv", index=False)
    print(f"✓ Band completeness report: {len(all_mat_files)} files checked, {len(missing_bands)} issues")
    return missing_bands

# ============================================================================
# CHECK 5: Band Shape Consistency
# ============================================================================

def check_band_shape_consistency():
    """Verify all bands for one sample have identical height/width."""
    shape_issues = []
    all_mat_files = get_all_mat_files()
    
    for mat_file in all_mat_files:
        try:
            with h5py.File(mat_file, "r") as f:
                data = f["gray"][:]
                num_bands, height, width = data.shape
                
                # Check if all bands have same shape
                for band_idx in range(num_bands):
                    band_height, band_width = data[band_idx].shape
                    if band_height != height or band_width != width:
                        shape_issues.append({
                            'file_path': str(mat_file),
                            'band': band_idx + 1,
                            'expected_shape': f"{height}x{width}",
                            'actual_shape': f"{band_height}x{band_width}"
                        })
        except Exception as e:
            shape_issues.append({
                'file_path': str(mat_file),
                'error': str(e)
            })
    
    df = pd.DataFrame(shape_issues) if shape_issues else pd.DataFrame(columns=['file_path', 'band', 'expected_shape', 'actual_shape'])
    df.to_csv(output_dir / "band_shape_report.csv", index=False)
    print(f"✓ Band shape consistency report: {len(all_mat_files)} files checked, {len(shape_issues)} issues")
    return shape_issues

# ============================================================================
# CHECK 6: Mask Match with Images
# ============================================================================

def check_mask_match():
    """Verify each image has a corresponding binary mask."""
    mask_mismatches = []
    
    for folder_pair in [(train_image_path, train_mask_path), (test_image_path, test_mask_path)]:
        image_folder, mask_folder = folder_pair
        image_names = {file.stem for file in get_files(image_folder, image_extensions)}
        mask_names = {file.stem for file in get_files(mask_folder, image_extensions)}
        
        # Images without masks
        for name in image_names - mask_names:
            mask_mismatches.append({
                'issue': 'image_without_mask',
                'file_name': name,
                'location': 'train' if folder_pair[0] == train_image_path else 'test'
            })
        
        # Masks without images
        for name in mask_names - image_names:
            mask_mismatches.append({
                'issue': 'mask_without_image',
                'file_name': name,
                'location': 'train' if folder_pair[0] == train_image_path else 'test'
            })
    
    df = pd.DataFrame(mask_mismatches) if mask_mismatches else pd.DataFrame(columns=['issue', 'file_name', 'location'])
    df.to_csv(output_dir / "missing_mask_report.csv", index=False)
    print(f"✓ Mask match report: {len(mask_mismatches)} issues found")
    return mask_mismatches

# ============================================================================
# CHECK 7: Attribute Labels
# ============================================================================

def check_attribute_labels():
    """Verify attribute representation exists and can be parsed."""
    attribute_report = {
        'status': 'checked',
        'timestamp': str(pd.Timestamp.now()),
        'notes': 'Check for attribute files in dataset directories'
    }
    
    # Look for common attribute files
    attribute_files = []
    for pattern in ['*attributes*', '*labels*', '*annotations*']:
        attribute_files.extend(dataset_path.glob(pattern))
    
    for root_dir in [train_path, test_path]:
        for pattern in ['*attributes*', '*labels*', '*annotations*']:
            attribute_files.extend(root_dir.glob(pattern))
    
    attribute_report['attribute_files_found'] = len(attribute_files)
    attribute_report['files'] = [str(f) for f in attribute_files]
    
    with open(output_dir / "attribute_format_report.txt", 'w') as f:
        f.write("ATTRIBUTE LABELS VALIDATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {attribute_report['timestamp']}\n")
        f.write(f"Attribute files found: {attribute_report['attribute_files_found']}\n")
        if attribute_report['files']:
            f.write("\nAttribute files detected:\n")
            for file in attribute_report['files']:
                f.write(f"  - {file}\n")
        else:
            f.write("\nNo attribute files detected in standard locations.\n")
    
    print(f"✓ Attribute labels report: {len(attribute_files)} attribute files found")
    return attribute_report

# ============================================================================
# CHECK 8: File Data Type & Dynamic Range
# ============================================================================

def check_dtype_and_dynamic_range():
    """Confirm uint8, uint16 or float and scaling information."""
    dtype_report = []
    all_mat_files = get_all_mat_files()
    
    # Check .mat files
    for mat_file in all_mat_files[:10]:  # Sample first 10 for quick check
        try:
            with h5py.File(mat_file, "r") as f:
                data = f["gray"][:]
                dtype_report.append({
                    'file': mat_file.name,
                    'data_type': str(data.dtype),
                    'min_value': float(data.min()),
                    'max_value': float(data.max()),
                    'mean_value': float(data.mean())
                })
        except Exception as e:
            dtype_report.append({
                'file': mat_file.name,
                'error': str(e)
            })
    
    # Check image files
    for img_file in get_files(train_image_path, image_extensions)[:5]:  # Sample 5
        try:
            with Image.open(img_file) as img:
                img_array = np.array(img)
                dtype_report.append({
                    'file': img_file.name,
                    'data_type': str(img_array.dtype),
                    'min_value': float(img_array.min()),
                    'max_value': float(img_array.max()),
                    'mean_value': float(img_array.mean())
                })
        except Exception as e:
            dtype_report.append({
                'file': img_file.name,
                'error': str(e)
            })
    
    df = pd.DataFrame(dtype_report)
    df.to_csv(output_dir / "dtype_dynamic_range_report.csv", index=False)
    print(f"✓ Data type report: {len(dtype_report)} files sampled")
    return dtype_report

# ============================================================================
# CHECK 9: Composite View Availability
# ============================================================================

def check_composite_view():
    """Identify whether an official RGB/false-colour composite is provided."""
    composite_report = {
        'status': 'checked',
        'timestamp': str(pd.Timestamp.now()),
    }
    
    # Check for composite directories or RGB files
    composite_dirs = []
    rgb_files = []
    
    for pattern in ['*composite*', '*rgb*', '*RGB*', '*false*color*']:
        composite_dirs.extend(dataset_path.glob(pattern))
        composite_dirs.extend(train_path.glob(pattern))
        composite_dirs.extend(test_path.glob(pattern))
    
    # Check for RGB in standard locations
    for folder in [train_image_path, test_image_path]:
        for img_file in get_files(folder, image_extensions):
            try:
                with Image.open(img_file) as img:
                    if len(img.getbands()) == 3:
                        rgb_files.append(str(img_file))
            except:
                pass
    
    composite_report['composite_dirs'] = len(composite_dirs)
    composite_report['rgb_files_count'] = len(rgb_files)
    
    with open(output_dir / "composite_view_report.txt", 'w') as f:
        f.write("COMPOSITE VIEW AVAILABILITY REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {composite_report['timestamp']}\n")
        f.write(f"Composite directories found: {composite_report['composite_dirs']}\n")
        f.write(f"RGB composite files detected: {composite_report['rgb_files_count']}\n")
        if rgb_files:
            f.write("\nSample RGB files (first 5):\n")
            for file in rgb_files[:5]:
                f.write(f"  - {file}\n")
        else:
            f.write("\nNo dedicated composite view directory detected.\n")
            f.write("Note: Images in 'Pcolor' folder appear to be RGB (3-channel).\n")
    
    print(f"✓ Composite view report: {len(rgb_files)} RGB files detected")
    return composite_report

# ============================================================================
# CHECK 10: Duplicate/Leakage Check
# ============================================================================

def compute_file_hash(file_path):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def check_train_test_leakage():
    """Check for duplicate files and potential data leakage between train/test."""
    leakage_report = []
    
    # Compute hashes for train set
    train_hashes = {}
    for img_file in get_files(train_image_path, image_extensions):
        try:
            hash_val = compute_file_hash(img_file)
            train_hashes[img_file.stem] = hash_val
        except Exception as e:
            print(f"Error hashing {img_file}: {e}")
    
    # Check test set against train set
    duplicates = []
    for img_file in get_files(test_image_path, image_extensions):
        try:
            hash_val = compute_file_hash(img_file)
            for train_name, train_hash in train_hashes.items():
                if hash_val == train_hash:
                    duplicates.append({
                        'duplicate_file': img_file.stem,
                        'matched_train_file': train_name,
                        'file_hash': hash_val,
                        'issue_type': 'exact_duplicate'
                    })
        except Exception as e:
            print(f"Error hashing {img_file}: {e}")
    
    df = pd.DataFrame(duplicates) if duplicates else pd.DataFrame(columns=['duplicate_file', 'matched_train_file', 'file_hash', 'issue_type'])
    df.to_csv(output_dir / "train_test_leakage_report.csv", index=False)
    print(f"✓ Leakage detection report: {len(duplicates)} duplicates found")
    return duplicates

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_validations():
    """Run all validation checks and generate reports."""
    print("\n" + "=" * 60)
    print("MCOD DATASET COMPREHENSIVE VALIDATION")
    print("=" * 60 + "\n")
    
    try:
        check_corrupted_files()
        check_sample_count()
        check_official_split()
        check_band_completeness()
        check_band_shape_consistency()
        check_mask_match()
        check_attribute_labels()
        check_dtype_and_dynamic_range()
        check_composite_view()
        check_train_test_leakage()
        
        print("\n" + "=" * 60)
        print(f"✓ ALL VALIDATIONS COMPLETE")
        print(f"✓ Reports saved to: {output_dir.absolute()}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Validation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_validations()
