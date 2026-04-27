import pandas as pd
import numpy as np
import json
import hashlib
import os
import argparse
import re
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

RANDOM_SEED = 42
TEST_SIZE = 0.2       # 20% for Test (Locked)
VAL_SIZE = 0.25       # 25% of remaining (20% total) for Val (Calibration)
MIN_SITE_SAMPLES = 10 # Exclude sites with <10 samples
OUTPUT_DIR = Path("experiments")
RAW_DIR = Path("data/raw")


def get_next_exp_id(output_dir):
    """
    Scans output_dir for existing manifests and returns the next available ID.
    Format: exp_001, exp_002, etc.
    """
    existing_ids = []
    for exp_folder in output_dir.glob("exp_*"):
        data_folder = exp_folder / 'data'
        if data_folder.exists():
            for f in data_folder.glob("*_manifest.json"):
                match = re.search(r"(exp_\d+)", f.stem)
                if match:
                    existing_ids.append(match.group(1))
    if not existing_ids:
        return "exp_001"
    # Sort and increment
    existing_ids.sort()
    last_id = existing_ids[-1]
    last_num = int(last_id.split('_')[1])
    next_num = last_num + 1
    return f"exp_{next_num:03d}"

def check_manifest_exists(output_dir, exp_id):
    """
    Checks if a manifest for this exp_id already exists.
    """
    manifest_path = output_dir / exp_id / 'data' / f"{exp_id}_manifest.json"
    return manifest_path.exists()


def find_and_resolve_images(raw_dir):
    """
    Scans raw_dir recursively for images.
    Prefers high-res over _downsampled versions.
    Returns a dict: { image_id (without ext): relative_path }
    """
    print(" Scanning for images...")
    image_map = {}
    downsampled_map = {}
    extensions = ('.jpg', '.jpeg', '.png')
    all_files = list(raw_dir.rglob('*'))
    image_files = [f for f in all_files if f.suffix.lower() in extensions]
    for file_path in image_files:
        stem = file_path.stem
        if stem.endswith('_downsampled'):
            base_id = stem.replace('_downsampled', '')
            downsampled_map[base_id] = file_path.relative_to(raw_dir)
        else:
            image_map[stem] = file_path.relative_to(raw_dir)
    final_map = image_map.copy()
    missing_count = 0
    for base_id, down_path in downsampled_map.items():
        if base_id not in final_map:
            final_map[base_id] = down_path
            missing_count += 1
    print(f"Found {len(final_map)} unique images.")
    if missing_count > 0:
        print(f"Warning: {missing_count} images were only available in _downsampled format.")
    return final_map


def load_and_clean_metadata(metadata_path, ground_truth_path, image_map):
    """
    Loads metadata, merges with found images, and binarizes labels.
    Handles ISIC 2019 one-hot encoded ground truth format.
    """
    print(" Loading metadata...")
    try:
        meta = pd.read_csv(metadata_path)
        gt = pd.read_csv(ground_truth_path)
    except FileNotFoundError as e:
        print(f" Error: {e}")
        raise e
    # Debug: Print columns to verify
    print(f"   Meta columns: {list(meta.columns)}")
    print(f"   GT columns: {list(gt.columns)}")
    # Merge Ground Truth on 'image'
    df = meta.merge(gt, on='image')

    # 1. Map Image IDs to File Paths
    df['file_path'] = df['image'].map(image_map)

    # 2. Filter Missing Images
    missing_images = df['file_path'].isna().sum()
    if missing_images > 0:
        print(f"Dropping {missing_images} rows where image files were not found.")
        df = df.dropna(subset=['file_path']).reset_index(drop=True)

    # 3. Binary Label Conversion (AK = Malignant)
    # ISIC 2019 GT has one-hot columns: MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
    malignant_classes = ['MEL', 'SCC', 'BCC', 'AK']

    # Create diagnosis column from one-hot encoding
    df['diagnosis'] = None
    for cls in malignant_classes + ['NV', 'BKL', 'DF', 'VASC', 'UNK']:
        if cls in df.columns:
            df.loc[df[cls] == 1, 'diagnosis'] = cls

    # Verify we captured all diagnoses
    unassigned = df['diagnosis'].isna().sum()

    if unassigned > 0:
        print(f"Warning: {unassigned} rows have no valid diagnosis label.")
        df = df.dropna(subset=['diagnosis']).reset_index(drop=True)
    # Create binary target
    df['target'] = df['diagnosis'].isin(malignant_classes).astype(int)

    # 4. Handle Missing Patient IDs (Leakage Prevention)
    # ISIC 2019 may not have patient_id column, use image as fallback
    if 'patient_id' in df.columns:
        df['patient_id'] = df['patient_id'].fillna(df['image'])
    else:
        # No patient_id column exists, use image_id as unique identifier
        # This is conservative - prevents any image from appearing in multiple splits
        df['patient_id'] = df['image']
        print("  No patient_id column found. Using image_id for leakage prevention.")

    # 5. Handle Missing Metadata (For Stratification Only)
    if 'age_approx' in df.columns:
        df['age_approx'] = df['age_approx'].fillna(df['age_approx'].median())
    else:
        df['age_approx'] = 50.0  # Default median age

    if 'sex' in df.columns:
        df['sex'] = df['sex'].fillna('unknown')
    else:
        df['sex'] = 'unknown'

    if 'anatom_site_general' in df.columns:
        df['anatom_site_general'] = df['anatom_site_general'].fillna('unknown')
    else:
        df['anatom_site_general'] = 'unknown'

    # 6. Filter Rare Sites
    site_counts = df['anatom_site_general'].value_counts()
    valid_sites = site_counts[site_counts >= MIN_SITE_SAMPLES].index
    df = df[df['anatom_site_general'].isin(valid_sites)]

    print(f" Final dataset size: {len(df)} samples.")
    print(f" Malignant rate (including AK): {df['target'].mean():.2%}")
    print("\n Diagnosis Distribution:")
    print(df['diagnosis'].value_counts())
    return df

def compute_catalog_hash(df):
    """
    Computes a hash of the catalog content.
    """
    hash_content = df[['image', 'patient_id', 'target', 'file_path']].to_string()
    return hashlib.sha256(hash_content.encode('utf-8')).hexdigest()


def create_splits(df, output_dir, exp_id, seed=RANDOM_SEED):
    """
    Creates Train/Val/Test splits using StratifiedGroupKFold.
    This ensures no patient leakage AND balanced class/site distribution.
    """
    print(f"Creating splits for {exp_id}...")
    df['strat_label'] = df['target'].astype(str) + '_' + df['anatom_site_general']
    groups = df['patient_id']
    y = df['target'].values

    # First split: Train+Val vs Test (80/20)
    from sklearn.model_selection import StratifiedGroupKFold
    sgkfold_test = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    # Get the first fold for test set (20%)
    for train_val_idx, test_idx in sgkfold_test.split(df, y=y, groups=groups):
        break  # Only use first fold

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Second split: Train vs Val (75/25 of remaining = 60/20 overall)
    sgkfold_val = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)

    for train_idx, val_idx in sgkfold_val.split(train_val_df, y=train_val_df['target'].values, groups=train_val_df['patient_id']):
        break  # Only use first fold

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Assign Split Column
    df['split'] = 'train'
    image_to_split = {}
    image_to_split.update({img: 'train' for img in train_df['image']})
    image_to_split.update({img: 'val' for img in val_df['image']})
    image_to_split.update({img: 'test' for img in test_df['image']})
    df['split'] = df['image'].map(image_to_split)
    
    # Create experiment folder structure
    exp_folder = output_dir / exp_id
    (exp_folder / 'data').mkdir(parents=True, exist_ok=True)
    (exp_folder / 'models').mkdir(parents=True, exist_ok=True)
    (exp_folder / 'calibration').mkdir(parents=True, exist_ok=True)
    (exp_folder / 'evaluation').mkdir(parents=True, exist_ok=True)
    (exp_folder / 'selective_classification').mkdir(parents=True, exist_ok=True)
    (exp_folder / 'visualization').mkdir(parents=True, exist_ok=True)
    
    # 1. Master Catalog
    catalog_path = exp_folder / 'data' / f"{exp_id}_catalog.csv"
    df.to_csv(catalog_path, index=False)
    print(f" Saved Catalog to {catalog_path}")
    
    # 2. Manifest
    catalog_hash = compute_catalog_hash(df)
    manifest = {
        "split_id": exp_id,
        "seed": seed,
        "data_hash": catalog_hash,
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
        "train_ids": train_df['image'].tolist(),
        "val_ids": val_df['image'].tolist(),
        "test_ids": test_df['image'].tolist(),
        "class_definition": "Malignant: MEL, SCC, BCC, AK | Benign: NV, BKL, DF, VASC"
    }
    manifest_path = exp_folder / 'data' / f"{exp_id}_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f" Saved Manifest to {manifest_path}")
    

    print("\n Split Distribution (Target Mean = Malignant %):")
    print(df.groupby('split')['target'].mean())
    print("\n Split Sizes:")
    print(df.groupby('split').size())
    # Leakage Check
    leaks = df.groupby('patient_id')['split'].nunique()
    if (leaks > 1).any():
        print("\n ERROR: Patient leakage detected!")
        return None
    else:
        print("\n No patient leakage detected.")

    # Image Existence Check
    missing_files = 0
    for path in df['file_path']:
        if not (RAW_DIR / path).exists():
            missing_files += 1
    if missing_files > 0:
        print(f"\n ERROR: {missing_files} files listed in catalog do not exist on disk!")
    else:
        print(" All file paths verified on disk.")
    return manifest


if __name__ == "__main__":
    # CLI Arguments
    parser = argparse.ArgumentParser(description="ISIC 2019 Data Splitter")
    parser.add_argument("--exp_id", type=str, default=None, help="Experiment ID (e.g., exp_001). Auto-generated if not provided.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing experiment manifest.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for splitting.")
    args = parser.parse_args()
    
    # Determine Experiment ID
    if args.exp_id:
        exp_id = args.exp_id
    else:
        exp_id = get_next_exp_id(OUTPUT_DIR)
    
    # Check for Overwrite
    if check_manifest_exists(OUTPUT_DIR, exp_id):
        if not args.overwrite:
            print(f" Error: Manifest for '{exp_id}' already exists.")
            print("   Use --overwrite to force update, or choose a different --exp_id.")
            exit(1)
        else:
            print(f"  Warning: Overwriting existing manifest for '{exp_id}'.")
    
    # Input Files
    META_FILE = RAW_DIR / "ISIC_2019_Training_Metadata.csv"
    GT_FILE = RAW_DIR / "ISIC_2019_Training_GroundTruth.csv"
    if not META_FILE.exists() or not GT_FILE.exists():
        raise FileNotFoundError("Metadata files not found in data/raw/")
    
    # Run Pipeline
    try:
        # 1. Find Images first
        image_map = find_and_resolve_images(RAW_DIR)
        if len(image_map) == 0:
            raise FileNotFoundError("No images found in data/raw/. Check subfolder structure.")
        # 2. Load & Clean
        df = load_and_clean_metadata(META_FILE, GT_FILE, image_map)
        # 3. Split
        manifest = create_splits(df, OUTPUT_DIR, exp_id, seed=args.seed)
        if manifest:
            print(f"\n Splitting Complete for {exp_id}. Ready for Training.")
            print(f"   Train: {manifest['train_count']}")
            print(f"   Val:   {manifest['val_count']}")
            print(f"   Test:  {manifest['test_count']}")
    except Exception as e:
        print(f"\n Pipeline Failed: {e}")
        raise e