"""
REST DATASETS COMPREHENSIVE AUDIT PIPELINE
- Audits CAMO-COCO-V.1.0, COD10K-v3, and NC4K
- RGB 3-channel datasets (no multispectral checks)
- Saves CSV reports + visualizations per dataset
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import hashlib
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import json
import warnings

warnings.filterwarnings("ignore")

# =========================
# HELPERS
# =========================

def files(folder, exts):
    """Get files with specific extensions"""
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.suffix.lower() in exts]

def sha256(path):
    """Calculate SHA256 hash of file"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def p_hash(img_path):
    """Calculate perceptual hash for near-duplicate detection"""
    try:
        img = Image.open(img_path).convert("L").resize((16,16))
        arr = np.array(img)
        avg = arr.mean()
        return ''.join('1' if x > avg else '0' for x in arr.flatten())
    except:
        return None

# =========================
# CAMO-COCO-V.1.0 AUDIT
# =========================

def audit_camo_coco():
    print("\n" + "="*60)
    print("AUDITING: CAMO-COCO-V.1.0")
    print("="*60)
    
    root = Path("data/CAMO-COCO-V.1.0/CAMO-COCO-V.1.0-CVIU2019")
    out = Path("data_audit/CAMO-COCO-V.1.0")
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    
    camo_img = root / "Camouflage/Images"
    camo_gt = root / "Camouflage/GT"
    noncamo_img = root / "Non-Camouflage/Images"
    noncamo_gt = root / "Non-Camouflage/GT"
    
    # 1. SAMPLE COUNT
    print("Computing sample counts...")
    data = {
        "category": ["Camouflage", "Non-Camouflage"],
        "images": [
            len(files(camo_img, {".jpg", ".png", ".tif"})),
            len(files(noncamo_img, {".jpg", ".png", ".tif"}))
        ],
        "masks": [
            len(files(camo_gt, {".jpg", ".png", ".tif"})),
            len(files(noncamo_gt, {".jpg", ".png", ".tif"}))
        ]
    }
    df_count = pd.DataFrame(data)
    df_count.to_csv(out / "sample_count.csv", index=False)
    print(f"  Sample counts: {df_count.to_dict('list')}")
    
    # 2. MASK INTEGRITY
    print("Checking mask integrity...")
    issues = []
    
    for category, img_dir, gt_dir in [("Camouflage", camo_img, camo_gt),
                                       ("Non-Camouflage", noncamo_img, noncamo_gt)]:
        img_stems = {f.stem for f in files(img_dir, {".png", ".jpg", ".tif"})}
        gt_stems = {f.stem for f in files(gt_dir, {".png", ".jpg", ".tif"})}
        
        for s in img_stems - gt_stems:
            issues.append((category, s, "missing_mask"))
        for s in gt_stems - img_stems:
            issues.append((category, s, "orphan_mask"))
    
    df_mask = pd.DataFrame(issues, columns=["category", "sample", "issue"])
    df_mask.to_csv(out / "mask_integrity_report.csv", index=False)
    print(f"  Mask integrity issues: {len(df_mask)}")
    
    # 3. RESOLUTION
    print("Analyzing resolutions...")
    rows = []
    for cat_name, img_dir in [("Camouflage", camo_img), ("Non-Camouflage", noncamo_img)]:
        for f in files(img_dir, {".png", ".jpg", ".tif"}):
            try:
                img = Image.open(f)
                w, h = img.size
                rows.append([cat_name, w, h, w*h])
            except:
                continue
    
    df_res = pd.DataFrame(rows, columns=["category", "width", "height", "area"])
    df_res.to_csv(out / "resolution_distribution.csv", index=False)
    print(f"  Resolutions: {len(df_res)} images analyzed")
    
    # 4. OBJECT AREA
    print("Computing object area ratios...")
    rows = []
    for cat_name, gt_dir in [("Camouflage", camo_gt), ("Non-Camouflage", noncamo_gt)]:
        for f in files(gt_dir, {".png", ".jpg", ".tif"}):
            try:
                arr = np.array(Image.open(f).convert("L"))
                ratio = (arr > 127).sum() / arr.size
                rows.append([cat_name, f.name, ratio])
            except:
                continue
    
    df_area = pd.DataFrame(rows, columns=["category", "file", "area_ratio"])
    df_area.to_csv(out / "object_area_distribution.csv", index=False)
    
    # Small objects
    small = df_area[df_area["area_ratio"] < 0.001]
    small.to_csv(out / "small_object_verification.csv", index=False)
    print(f"  Small objects (<0.1%): {len(small)}")
    
    # 5. DUPLICATES (EXACT)
    print("Detecting exact duplicates...")
    imgs = files(camo_img, {".png", ".jpg", ".tif"}) + files(noncamo_img, {".png", ".jpg", ".tif"})
    hmap = defaultdict(list)
    
    for i in imgs:
        hmap[sha256(i)].append(i.name)
    
    dups = {k: v for k, v in hmap.items() if len(v) > 1}
    df_dup = pd.DataFrame([(k, v) for k, v in dups.items()], columns=["hash", "files"])
    df_dup.to_csv(out / "duplicate_exact_report.csv", index=False)
    print(f"  Exact duplicates: {len(dups)} groups")
    
    # 6. NEAR DUPLICATES
    print("Detecting near duplicates...")
    hashes = {}
    for i in imgs:
        h = p_hash(i)
        if h:
            hashes[i.name] = h
    
    def dist(a, b):
        return sum(x != y for x, y in zip(a, b))
    
    pairs = []
    keys = list(hashes.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            d = dist(hashes[keys[i]], hashes[keys[j]])
            if d < 10:
                pairs.append([keys[i], keys[j], d])
    
    df_ndups = pd.DataFrame(pairs, columns=["img1", "img2", "distance"])
    df_ndups.to_csv(out / "duplicate_near_report.csv", index=False)
    print(f"  Near duplicates: {len(pairs)} pairs")
    
    # 7. SAMPLE VISUALIZATION
    print("Creating sample visualizations...")
    camo_imgs = files(camo_img, {".png", ".jpg", ".tif"})
    noncamo_imgs = files(noncamo_img, {".png", ".jpg", ".tif"})
    
    if camo_imgs and noncamo_imgs:
        fig_obj, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        try:
            img_c = Image.open(camo_imgs[0])
            axes[0].imshow(img_c)
            axes[0].set_title("Camouflage Sample")
            axes[0].axis("off")
        except:
            pass
        
        try:
            img_nc = Image.open(noncamo_imgs[0])
            axes[1].imshow(img_nc)
            axes[1].set_title("Non-Camouflage Sample")
            axes[1].axis("off")
        except:
            pass
        
        plt.tight_layout()
        plt.savefig(fig / "sample_visualization.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"✓ CAMO-COCO-V.1.0 audit complete. Results: {out.resolve()}\n")

# =========================
# COD10K-v3 AUDIT
# =========================

def audit_cod10k():
    print("\n" + "="*60)
    print("AUDITING: COD10K-v3")
    print("="*60)
    
    root = Path("data/COD10K-v3")
    out = Path("data_audit/COD10K-v3")
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    
    train_img = root / "Train/Image"
    test_img = root / "Test/Image"
    train_gt_obj = root / "Train/GT_Object"
    test_gt_obj = root / "Test/GT_Object"
    
    # 1. SAMPLE COUNT
    print("Computing sample counts...")
    data = {
        "split": ["train", "test"],
        "images": [
            len(files(train_img, {".jpg", ".png", ".tif"})),
            len(files(test_img, {".jpg", ".png", ".tif"}))
        ],
        "gt_object": [
            len(files(train_gt_obj, {".jpg", ".png", ".tif"})),
            len(files(test_gt_obj, {".jpg", ".png", ".tif"}))
        ]
    }
    df_count = pd.DataFrame(data)
    df_count.to_csv(out / "sample_count.csv", index=False)
    print(f"  Sample counts: {df_count.to_dict('list')}")
    
    # 2. MASK INTEGRITY (GT_Object)
    print("Checking mask integrity...")
    issues = []
    
    for split, img_dir, gt_dir in [("train", train_img, train_gt_obj),
                                    ("test", test_img, test_gt_obj)]:
        img_stems = {f.stem for f in files(img_dir, {".png", ".jpg", ".tif"})}
        gt_stems = {f.stem for f in files(gt_dir, {".png", ".jpg", ".tif"})}
        
        for s in img_stems - gt_stems:
            issues.append((split, s, "missing_gt_object"))
        for s in gt_stems - img_stems:
            issues.append((split, s, "orphan_gt_object"))
    
    df_mask = pd.DataFrame(issues, columns=["split", "sample", "issue"])
    df_mask.to_csv(out / "mask_integrity_report.csv", index=False)
    print(f"  Mask integrity issues: {len(df_mask)}")
    
    # 3. RESOLUTION
    print("Analyzing resolutions...")
    rows = []
    for split_name, img_dir in [("train", train_img), ("test", test_img)]:
        for f in files(img_dir, {".png", ".jpg", ".tif"}):
            try:
                img = Image.open(f)
                w, h = img.size
                rows.append([split_name, w, h, w*h])
            except:
                continue
    
    df_res = pd.DataFrame(rows, columns=["split", "width", "height", "area"])
    df_res.to_csv(out / "resolution_distribution.csv", index=False)
    print(f"  Resolutions: {len(df_res)} images analyzed")
    
    # 4. OBJECT AREA
    print("Computing object area ratios...")
    rows = []
    for split_name, gt_dir in [("train", train_gt_obj), ("test", test_gt_obj)]:
        for f in files(gt_dir, {".png", ".jpg", ".tif"}):
            try:
                arr = np.array(Image.open(f).convert("L"))
                ratio = (arr > 127).sum() / arr.size
                rows.append([split_name, f.name, ratio])
            except:
                continue
    
    df_area = pd.DataFrame(rows, columns=["split", "file", "area_ratio"])
    df_area.to_csv(out / "object_area_distribution.csv", index=False)
    
    # Small objects
    small = df_area[df_area["area_ratio"] < 0.001]
    small.to_csv(out / "small_object_verification.csv", index=False)
    print(f"  Small objects (<0.1%): {len(small)}")
    
    # 5. DUPLICATES (EXACT)
    print("Detecting exact duplicates...")
    imgs = files(train_img, {".png", ".jpg", ".tif"}) + files(test_img, {".png", ".jpg", ".tif"})
    hmap = defaultdict(list)
    
    for i in imgs:
        hmap[sha256(i)].append(i.name)
    
    dups = {k: v for k, v in hmap.items() if len(v) > 1}
    df_dup = pd.DataFrame([(k, v) for k, v in dups.items()], columns=["hash", "files"])
    df_dup.to_csv(out / "duplicate_exact_report.csv", index=False)
    print(f"  Exact duplicates: {len(dups)} groups")
    
    # 6. NEAR DUPLICATES
    print("Detecting near duplicates...")
    hashes = {}
    for i in imgs:
        h = p_hash(i)
        if h:
            hashes[i.name] = h
    
    def dist(a, b):
        return sum(x != y for x, y in zip(a, b))
    
    pairs = []
    keys = list(hashes.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            d = dist(hashes[keys[i]], hashes[keys[j]])
            if d < 10:
                pairs.append([keys[i], keys[j], d])
    
    df_ndups = pd.DataFrame(pairs, columns=["img1", "img2", "distance"])
    df_ndups.to_csv(out / "duplicate_near_report.csv", index=False)
    print(f"  Near duplicates: {len(pairs)} pairs")
    
    # 7. SAMPLE VISUALIZATION
    print("Creating sample visualizations...")
    train_imgs = files(train_img, {".png", ".jpg", ".tif"})
    test_imgs = files(test_img, {".png", ".jpg", ".tif"})
    
    if train_imgs and test_imgs:
        fig_obj, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        try:
            img_tr = Image.open(train_imgs[0])
            axes[0].imshow(img_tr)
            axes[0].set_title("Train Sample")
            axes[0].axis("off")
        except:
            pass
        
        try:
            img_te = Image.open(test_imgs[0])
            axes[1].imshow(img_te)
            axes[1].set_title("Test Sample")
            axes[1].axis("off")
        except:
            pass
        
        plt.tight_layout()
        plt.savefig(fig / "sample_visualization.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"✓ COD10K-v3 audit complete. Results: {out.resolve()}\n")

# =========================
# NC4K AUDIT
# =========================

def audit_nc4k():
    print("\n" + "="*60)
    print("AUDITING: NC4K")
    print("="*60)
    
    root = Path("data/NC4K")
    out = Path("data_audit/NC4K")
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    
    img_dir = root / "Imgs"
    gt_dir = root / "GT"
    inst_dir = root / "Instance"
    
    # 1. SAMPLE COUNT
    print("Computing sample counts...")
    data = {
        "type": ["images", "gt", "instance"],
        "count": [
            len(files(img_dir, {".jpg", ".png", ".tif"})),
            len(files(gt_dir, {".jpg", ".png", ".tif"})),
            len(files(inst_dir, {".jpg", ".png", ".tif"}))
        ]
    }
    df_count = pd.DataFrame(data)
    df_count.to_csv(out / "sample_count.csv", index=False)
    print(f"  Sample counts: {df_count.to_dict('list')}")
    
    # 2. MASK INTEGRITY (both GT and Instance)
    print("Checking mask integrity...")
    issues = []
    
    img_stems = {f.stem for f in files(img_dir, {".png", ".jpg", ".tif"})}
    gt_stems = {f.stem for f in files(gt_dir, {".png", ".jpg", ".tif"})}
    inst_stems = {f.stem for f in files(inst_dir, {".png", ".jpg", ".tif"})}
    
    for s in img_stems - gt_stems:
        issues.append((s, "missing_gt"))
    for s in gt_stems - img_stems:
        issues.append((s, "orphan_gt"))
    for s in img_stems - inst_stems:
        issues.append((s, "missing_instance"))
    for s in inst_stems - img_stems:
        issues.append((s, "orphan_instance"))
    
    df_mask = pd.DataFrame(issues, columns=["sample", "issue"])
    df_mask.to_csv(out / "mask_integrity_report.csv", index=False)
    print(f"  Mask integrity issues: {len(df_mask)}")
    
    # 3. RESOLUTION
    print("Analyzing resolutions...")
    rows = []
    for f in files(img_dir, {".png", ".jpg", ".tif"}):
        try:
            img = Image.open(f)
            w, h = img.size
            rows.append([w, h, w*h])
        except:
            continue
    
    df_res = pd.DataFrame(rows, columns=["width", "height", "area"])
    df_res.to_csv(out / "resolution_distribution.csv", index=False)
    print(f"  Resolutions: {len(df_res)} images analyzed")
    
    # 4. OBJECT AREA
    print("Computing object area ratios...")
    rows = []
    for f in files(gt_dir, {".png", ".jpg", ".tif"}):
        try:
            arr = np.array(Image.open(f).convert("L"))
            ratio = (arr > 127).sum() / arr.size
            rows.append([f.name, ratio])
        except:
            continue
    
    df_area = pd.DataFrame(rows, columns=["file", "area_ratio"])
    df_area.to_csv(out / "object_area_distribution.csv", index=False)
    
    # Small objects
    small = df_area[df_area["area_ratio"] < 0.001]
    small.to_csv(out / "small_object_verification.csv", index=False)
    print(f"  Small objects (<0.1%): {len(small)}")
    
    # 5. DUPLICATES (EXACT)
    print("Detecting exact duplicates...")
    imgs = files(img_dir, {".png", ".jpg", ".tif"})
    hmap = defaultdict(list)
    
    for i in imgs:
        hmap[sha256(i)].append(i.name)
    
    dups = {k: v for k, v in hmap.items() if len(v) > 1}
    df_dup = pd.DataFrame([(k, v) for k, v in dups.items()], columns=["hash", "files"])
    df_dup.to_csv(out / "duplicate_exact_report.csv", index=False)
    print(f"  Exact duplicates: {len(dups)} groups")
    
    # 6. NEAR DUPLICATES
    print("Detecting near duplicates...")
    hashes = {}
    for i in imgs:
        h = p_hash(i)
        if h:
            hashes[i.name] = h
    
    def dist(a, b):
        return sum(x != y for x, y in zip(a, b))
    
    pairs = []
    keys = list(hashes.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            d = dist(hashes[keys[i]], hashes[keys[j]])
            if d < 10:
                pairs.append([keys[i], keys[j], d])
    
    df_ndups = pd.DataFrame(pairs, columns=["img1", "img2", "distance"])
    df_ndups.to_csv(out / "duplicate_near_report.csv", index=False)
    print(f"  Near duplicates: {len(pairs)} pairs")
    
    # 7. SAMPLE VISUALIZATION
    print("Creating sample visualizations...")
    imgs_list = files(img_dir, {".png", ".jpg", ".tif"})
    
    if imgs_list:
        fig_obj, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        try:
            img_sample = Image.open(imgs_list[0])
            axes[0].imshow(img_sample)
            axes[0].set_title("Image Sample")
            axes[0].axis("off")
        except:
            pass
        
        try:
            gt_sample = Image.open(list(files(gt_dir, {".png", ".jpg", ".tif"}))[0])
            axes[1].imshow(gt_sample, cmap="gray")
            axes[1].set_title("GT Sample")
            axes[1].axis("off")
        except:
            pass
        
        plt.tight_layout()
        plt.savefig(fig / "sample_visualization.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"✓ NC4K audit complete. Results: {out.resolve()}\n")

# =========================
# MAIN
# =========================

def run():
    print("\n" + "="*60)
    print("REST DATASETS COMPREHENSIVE AUDIT PIPELINE")
    print("="*60)
    
    audit_camo_coco()
    audit_cod10k()
    audit_nc4k()
    
    print("\n" + "="*60)
    print("ALL AUDITS COMPLETE")
    print("Results saved to: data_audit/")
    print("="*60 + "\n")

if __name__ == "__main__":
    run()
