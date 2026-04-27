import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from netcal.presentation import ReliabilityDiagram
from src.utils.cache import InferenceCache

def main(args):
    # Load configs
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.calibration) as f:
        calib = json.load(f)
    temperature = calib['temperature']
    print(f"🌡️  Temperature: {temperature:.4f}")
    
    # Get experiment ID for cache directory
    exp_id = manifest['split_id']  # e.g., 'exp_001'
    
    # Initialise cache with experiment-specific directory
    cache = InferenceCache(cache_dir=Path(args.cache_dir) / exp_id)
    
    # Load UNCALIBRATED cached inference
    print("\n📈 Loading UNCALIBRATED cached predictions...")
    probs_uncal, targets, _, _ = cache.load(
        args.checkpoint, args.manifest, 'val', temperature=None
    )
    if probs_uncal is None:
        print("No cached inference found! Run run_inference.py first.")
        print("   python src/run_inference.py --catalog ... --manifest ... --checkpoint ... --splits val,test")
        return
    
    # Load CALIBRATED cached inference (or apply temperature)
    print("Loading CALIBRATED cached predictions...")
    probs_cal, _, _, _ = cache.load(
        args.checkpoint, args.manifest, 'val', temperature=temperature
    )
    if probs_cal is None:
        # Apply temperature to uncalibrated probs
        print("   Applying temperature to cached predictions...")
        logits = torch.logit(torch.from_numpy(probs_uncal).clip(1e-7, 1-1e-7))
        probs_cal = torch.sigmoid(logits / temperature).numpy()
    
    # Calculate Brier Scores
    brier_uncal = brier_score_loss(targets, probs_uncal)
    brier_cal = brier_score_loss(targets, probs_cal)
    
    # Create output directory in experiment folder structure
    results_dir = Path(args.output_dir) / exp_id / 'visualization'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot UNCALIBRATED
    print("\n Creating UNCALIBRATED reliability diagram...")
    probs_uncal_2class = np.stack([1 - probs_uncal, probs_uncal], axis=1)
    fig_uncal = plt.figure(figsize=(10, 12))
    rd_uncal = ReliabilityDiagram(bins=10, metric='ECE')
    rd_uncal.plot(probs_uncal_2class, targets.astype(int))
    plt.suptitle(f'uncalibrated Brier Score: {brier_uncal:.4f}',
                 fontsize=11, fontweight='bold', y=1.02)
    fig_path_uncal = results_dir / 'reliability_diagram_uncalibrated.png'
    plt.savefig(fig_path_uncal, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved to {fig_path_uncal}")
    
    # Plot CALIBRATED
    print("Creating CALIBRATED reliability diagram...")
    probs_cal_2class = np.stack([1 - probs_cal, probs_cal], axis=1)
    fig_cal = plt.figure(figsize=(10, 12))
    rd_cal = ReliabilityDiagram(bins=10, metric='ECE')
    rd_cal.plot(probs_cal_2class, targets.astype(int))
    plt.suptitle(f'Calibrated (T={temperature:.2f}) Brier Score: {brier_cal:.4f}',
                 fontsize=11, fontweight='bold', y=1.02)
    fig_path_cal = results_dir / 'reliability_diagram_calibrated.png'
    plt.savefig(fig_path_cal, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {fig_path_cal}")
    
    # Print summary
    improvement = brier_uncal - brier_cal
    improvement_pct = (improvement / brier_uncal) * 100 if brier_uncal > 0 else 0
    print(f"\n{'='*50}")
    print(f"CALIBRATION SUMMARY")
    print(f"{'='*50}")
    print(f"Uncalibrated Brier Score: {brier_uncal:.4f}")
    print(f"Calibrated Brier Score:   {brier_cal:.4f}")
    print(f"Improvement:              {improvement:.4f} ({improvement_pct:.1f}% reduction)")
    print(f"{'='*50}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Calibration (Uses Cached Inference)')
    parser.add_argument('--catalog', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--calibration', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--output_dir', type=str, default='experiments')
    args = parser.parse_args()
    main(args)