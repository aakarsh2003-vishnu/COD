"""
MCOD DATASET COMPREHENSIVE AUDIT PIPELINE (FULL VERSION)
- Covers all audit items
- Optimized for large datasets
- Saves CSV reports + visualization
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import hashlib
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================

ROOT = Path("data/MCOD_resized")

TRAIN_IMG = ROOT / "TrainDataset/Pcolor"
TEST_IMG  = ROOT / "TestDataset/Pcolor"

TRAIN_MASK = ROOT / "TrainDataset/GT"
TEST_MASK  = ROOT / "TestDataset/GT"

TRAIN_MAT = ROOT / "TrainDataset/Mat"
TEST_MAT  = ROOT / "TestDataset/Mat"

OUT = Path("data_audit")
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# =========================
# HELPERS
# =========================

def files(folder, exts):
    if not folder.exists():
        return []
    return [f for f in folder.iterdir() if f.suffix.lower() in exts]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def p_hash(img_path):
    try:
        img = Image.open(img_path).convert("L").resize((16,16))
        arr = np.array(img)
        avg = arr.mean()
        return ''.join('1' if x > avg else '0' for x in arr.flatten())
    except:
        return None

# =========================
# 1. SAMPLE COUNT
# =========================

def sample_count():
    data = {
        "train_images": len(files(TRAIN_IMG,{".jpg",".png",".tif"})),
        "test_images": len(files(TEST_IMG,{".jpg",".png",".tif"})),
        "train_masks": len(files(TRAIN_MASK,{".jpg",".png",".tif"})),
        "test_masks": len(files(TEST_MASK,{".jpg",".png",".tif"})),
        "train_mat": len(files(TRAIN_MAT,{".mat"})),
        "test_mat": len(files(TEST_MAT,{".mat"})),
    }

    df = pd.DataFrame(data.items(), columns=["type","count"])
    df.to_csv(OUT/"dataset_summary.csv", index=False)
    return df

# =========================
# 2. MASK INTEGRITY
# =========================

def mask_integrity():
    train_img = {f.stem for f in files(TRAIN_IMG,{".png",".jpg",".tif"})}
    train_msk = {f.stem for f in files(TRAIN_MASK,{".png",".jpg",".tif"})}

    test_img = {f.stem for f in files(TEST_IMG,{".png",".jpg",".tif"})}
    test_msk = {f.stem for f in files(TEST_MASK,{".png",".jpg",".tif"})}

    issues = []

    for s in train_img - train_msk:
        issues.append(("train", s, "missing_mask"))
    for s in train_msk - train_img:
        issues.append(("train", s, "orphan_mask"))
    for s in test_img - test_msk:
        issues.append(("test", s, "missing_mask"))
    for s in test_msk - test_img:
        issues.append(("test", s, "orphan_mask"))

    df = pd.DataFrame(issues, columns=["split","sample","issue"])
    df.to_csv(OUT/"mask_integrity_report.csv", index=False)
    return df

# =========================
# 3. BAND INTEGRITY (S1–S8)
# =========================

def band_integrity():
    mats = files(TRAIN_MAT,{".mat"}) + files(TEST_MAT,{".mat"})

    issues = []

    for m in mats:
        try:
            with h5py.File(m,"r") as f:
                data = f["gray"]
                if data.shape[0] != 8:
                    issues.append((m.name, data.shape[0]))
        except:
            issues.append((m.name, "error"))

    df = pd.DataFrame(issues, columns=["file","bands"])
    df.to_csv(OUT/"band_integrity_report.csv", index=False)
    return df

# =========================
# 4. RESOLUTION ANALYSIS
# =========================

def resolution():
    rows = []

    for folder in [TRAIN_IMG, TEST_IMG, TRAIN_MASK, TEST_MASK]:
        for f in files(folder,{".png",".jpg",".tif"}):
            try:
                img = Image.open(f)
                w,h = img.size
                rows.append([folder.name, w, h, w*h])
            except:
                continue

    df = pd.DataFrame(rows, columns=["source","w","h","area"])
    df.to_csv(OUT/"resolution_distribution.csv", index=False)
    return df

# =========================
# 5. BAND STATISTICS (GLOBAL)
# =========================

def band_stats():
    mats = files(TRAIN_MAT,{".mat"}) + files(TEST_MAT,{".mat"})

    stats = []

    for b in range(8):
        vals = []

        for m in mats:
            try:
                with h5py.File(m,"r") as f:
                    band = f["gray"][b]
                    vals.append([band.mean(), band.std()])
            except:
                continue

        vals = np.array(vals)
        stats.append([
            f"S{b+1}",
            vals[:,0].mean() if len(vals) else 0,
            vals[:,1].mean() if len(vals) else 0
        ])

    df = pd.DataFrame(stats, columns=["band","mean","std"])
    df.to_csv(OUT/"band_statistics.csv", index=False)
    return df

# =========================
# 6. OBJECT AREA
# =========================

def object_area():
    masks = files(TRAIN_MASK,{".png",".jpg",".tif"}) + files(TEST_MASK,{".png",".jpg",".tif"})

    rows = []

    for m in masks:
        try:
            arr = np.array(Image.open(m).convert("L"))
            ratio = (arr > 127).sum() / arr.size
            rows.append([m.name, ratio])
        except:
            continue

    df = pd.DataFrame(rows, columns=["file","area_ratio"])
    df.to_csv(OUT/"object_area_distribution.csv", index=False)
    return df

# =========================
# 7. SMALL OBJECT CHECK
# =========================

def small_objects(df):
    small = df[df["area_ratio"] < 0.001]
    small.to_csv(OUT/"small_object_verification.csv", index=False)
    return small

# =========================
# 8. ATTRIBUTE DISTRIBUTION (PLACEHOLDER SAFE)
# =========================

def attribute_distribution():
    # MCOD does not always include structured attributes in MAT
    mats = files(TRAIN_MAT,{".mat"}) + files(TEST_MAT,{".mat"})

    counter = Counter()

    for m in mats:
        try:
            with h5py.File(m,"r") as f:
                keys = list(f.keys())
                counter.update(keys)
        except:
            continue

    df = pd.DataFrame(counter.items(), columns=["attribute","count"])
    df.to_csv(OUT/"attribute_distribution.csv", index=False)
    return df

# =========================
# 9. ATTRIBUTE CO-OCCURRENCE
# =========================

def attribute_cooccurrence():
    mats = files(TRAIN_MAT,{".mat"}) + files(TEST_MAT,{".mat"})

    pairs = Counter()

    for m in mats:
        try:
            with h5py.File(m,"r") as f:
                keys = list(f.keys())
                for i in range(len(keys)):
                    for j in range(i+1,len(keys)):
                        pairs[(keys[i],keys[j])] += 1
        except:
            continue

    df = pd.DataFrame([(a,b,c) for (a,b),c in pairs.items()],
                      columns=["attr1","attr2","count"])
    df.to_csv(OUT/"attribute_cooccurrence.csv", index=False)
    return df

# =========================
# 10. SPLIT DISTRIBUTION
# =========================

def split_distribution(area_df):
    train = area_df.iloc[:len(area_df)//2]["area_ratio"]
    test = area_df.iloc[len(area_df)//2:]["area_ratio"]

    df = pd.DataFrame({
        "metric":["mean","std","min","max"],
        "train":[train.mean(),train.std(),train.min(),train.max()],
        "test":[test.mean(),test.std(),test.min(),test.max()]
    })

    df.to_csv(OUT/"split_distribution_check.csv", index=False)
    return df

# =========================
# 11. DUPLICATES (EXACT)
# =========================

def duplicates_exact():
    imgs = files(TRAIN_IMG,{".png",".jpg",".tif"}) + files(TEST_IMG,{".png",".jpg",".tif"})

    hmap = defaultdict(list)

    for i in imgs:
        hmap[sha256(i)].append(i.name)

    dups = {k:v for k,v in hmap.items() if len(v)>1}

    df = pd.DataFrame([(k,v) for k,v in dups.items()],
                      columns=["hash","files"])
    df.to_csv(OUT/"duplicate_exact_report.csv", index=False)
    return df

# =========================
# 12. NEAR DUPLICATES
# =========================

def duplicates_near():
    imgs = files(TRAIN_IMG,{".png",".jpg",".tif"}) + files(TEST_IMG,{".png",".jpg",".tif"})

    hashes = {}

    for i in imgs:
        h = p_hash(i)
        if h:
            hashes[i.name] = h

    def dist(a,b):
        return sum(x!=y for x,y in zip(a,b))

    pairs = []

    keys = list(hashes.keys())

    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            d = dist(hashes[keys[i]], hashes[keys[j]])
            if d < 10:
                pairs.append([keys[i], keys[j], d])

    df = pd.DataFrame(pairs, columns=["img1","img2","distance"])
    df.to_csv(OUT/"duplicate_near_report.csv", index=False)
    return df

# =========================
# 13. VISUALIZATION (SIMPLE)
# =========================

def visualize_sample():
    mats = files(TRAIN_MAT,{".mat"})[:1]

    if not mats:
        return

    m = mats[0]

    with h5py.File(m,"r") as f:
        data = f["gray"][:]

    rgb = np.stack([data[2],data[1],data[0]],axis=2)
    rgb = (rgb - rgb.min())/(rgb.max()-rgb.min()+1e-8)

    plt.figure(figsize=(10,4))
    plt.imshow(rgb)
    plt.title("MCOD Sample Composite")
    plt.axis("off")
    plt.savefig(FIG/"sample_band_visualization.png")
    plt.close()

# =========================
# MAIN
# =========================

def run():
    print("\n=== MCOD FULL AUDIT START ===\n")

    sample_count()
    mask_integrity()
    band_integrity()
    res = resolution()
    band_stats()
    area = object_area()
    small_objects(area)
    attribute_distribution()
    attribute_cooccurrence()
    split_distribution(area)
    duplicates_exact()
    duplicates_near()
    visualize_sample()

    print("\n=== AUDIT COMPLETE ===")
    print(f"Saved in: {OUT.resolve()}")

if __name__ == "__main__":
    run()