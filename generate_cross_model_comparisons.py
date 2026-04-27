import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import platform

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.cache import InferenceCache
from src.analysis.core_metrics import (
    find_optimal_thresholds,
    apply_selective_thresholds,
    compute_confusion_matrix_3x2,
    compute_rates_from_confusion,
    generate_cost_coverage_curve
)

MAX_WORKERS = 28

COST_CONFIGS = {
    'Baseline': {'cFN': 1, 'cFP': 1, 'C_ref': 0.4},
    '2:1 cost of FN:FP': {'cFN': 2, 'cFP': 1, 'C_ref': 0.4},
    '4:1 cost of FN:FP': {'cFN': 4, 'cFP': 1, 'C_ref': 0.4},
    '10:1 cost of FN:FP': {'cFN': 10, 'cFP': 1, 'C_ref': 0.4},
    '20:1 cost of FN:FP': {'cFN': 20, 'cFP': 1, 'C_ref': 0.4},
}

MODEL_COLORS = {
    'exp_r50': '#1f77b4',      # Blue
    'exp_r101': '#ff7f0e',     # Orange
    'exp_r152': '#2ca02c',     # Green
    'exp_d121': '#d62728',     # Red
    'exp_d169': '#9467bd',     # Purple
    'exp_vb16': '#8c564b',     # Brown
    'exp_ct': '#e377c2',       # Pink
}

MODEL_NAMES = {
    'exp_r50': 'ResNet-50',
    'exp_r101': 'ResNet-101',
    'exp_r152': 'ResNet-152',
    'exp_d121': 'DenseNet-121',
    'exp_d169': 'DenseNet-169',
    'exp_vb16': 'ViT-B/16',
    'exp_ct': 'ConvNeXt-Tiny',
}

def _load_model_data(args):
    """Worker: Load one model's cache."""
    exp_folder, cache_base_dir, manifest_path, calib_path, split = args
    
    model_name = exp_folder.name
    try:
        ckpts = list((exp_folder / 'models').glob('*.pth'))
        if not ckpts:
            return {'error': f'No .pth in {exp_folder}/models', 'model': model_name}
        
        with open(calib_path) as f:
            calib = json.load(f)
        temperature = calib.get('temperature', 1.0)
        
        # Initialize cache with experiment-specific subfolder
        cache = InferenceCache(cache_dir=Path(cache_base_dir) / model_name)
        probs, targets, image_ids, _ = cache.load(str(ckpts[0]), str(manifest_path), split, temperature=temperature)
        
        if probs is None:
            return {'error': f'No cache for {model_name}', 'model': model_name}
        
        return {
            'model': model_name,
            'probs': probs,
            'targets': targets,
            'prevalence': float(np.mean(targets)),
            'n_samples': len(probs)
        }
    except Exception as e:
        return {'error': str(e), 'model': model_name}

def _compute_single_metric(args):
    """
    Worker: Compute metrics AND cost-coverage curve for (model, config) pair.
    """
    model_name, config_name, probs, targets, cost_fn, cost_fp, C_ref = args
    
    try:
        # 1. Compute point metrics for this config
        tau1, tau2 = find_optimal_thresholds(cost_fn, cost_fp, C_ref)
        
        if tau1 >= tau2:
            tau1, tau2 = 0.5, 0.5
            decisions = (probs >= 0.5).astype(int)
        else:
            decisions = apply_selective_thresholds(probs, tau1, tau2)
        
        cm = compute_confusion_matrix_3x2(targets, probs, tau1, tau2)
        rates = compute_rates_from_confusion(cm, normalize='by_class')
        
        n = len(targets)
        rejected = np.sum(decisions == 2)
        coverage = 1.0 - (rejected / n)
        
        classified_mask = decisions != 2
        n_cls = np.sum(classified_mask)
        accuracy = np.sum(decisions[classified_mask] == targets[classified_mask]) / n_cls if n_cls > 0 else 0.0
        
        # 2. Generate full cost-coverage curve (rejects one sample at a time)
        # This generates the smooth curve data
        df = generate_cost_coverage_curve(
            probs, targets, cost_fn, cost_fp, C_ref,
            use_baseline_norm=True
        )
        
        return {
            'model': model_name,
            'config': config_name,
            'coverage': coverage,
            'reject_rate': rejected / n,
            'sensitivity': rates['TPR'],
            'specificity': rates['TNR'],
            'accuracy': accuracy,
            'cost_curve_df': df,  # Store dataframe for plotting
            'success': True
        }
    except Exception as e:
        return {
            'model': model_name,
            'config': config_name,
            'error': str(e),
            'success': False
        }


def plot_cost_coverage_comparison(all_results: Dict, config_name: str, output_dir: Path):
    """
    Plot cost-coverage curves for all models.
    Uses ax.plot() for smooth continuous lines (matching run_unified_analysis.py).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_name, data in sorted(all_results.items()):
        if not data.get('success', False):
            continue
            
        color = MODEL_COLORS.get(model_name, 'gray')
        df = data['cost_curve_df']
        
        ax.plot(df['coverage'], df['avg_true_cost_removed_norm'],
                label=MODEL_NAMES.get(model_name, model_name),
                linewidth=2.0, color=color, alpha=0.8)
        
    
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Coverage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average True Cost of Classified Samples Relative to Baseline (No Rejection)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cost-Coverage Trade-off Across Models\n{config_name}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(ncol=2, fontsize=10, framealpha=0.9, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    safe_name = config_name.replace(':', '_').replace('/', '_').replace(' ', '_')
    plt.savefig(output_dir / f'cross_model_cost_coverage_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved cost-coverage comparison for {config_name}")

def plot_metrics_comparison(all_results: Dict, config_name: str, output_dir: Path):
    """Plot bar chart comparing metrics across all models."""
    metrics_to_plot = ['sensitivity', 'specificity', 'coverage', 'accuracy']
    titles = ['Sensitivity', 'Specificity', 'Coverage', 'Accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    models = sorted([m for m, d in all_results.items() if d.get('success', False)])
    x = np.arange(len(models))
    width = 0.6
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        values = [all_results[m][metric] for m in models]
        
        colors = [MODEL_COLORS.get(m, 'gray') for m in models]
        bars = ax.bar(x, values, width, color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} at {config_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAMES.get(m, m) for m in models], rotation=25, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(0.75, 1.0)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.suptitle(f'Cross-Model Performance Comparison\n{config_name}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    safe_name = config_name.replace(':', '_').replace('/', '_').replace(' ', '_')
    plt.savefig(output_dir / f'cross_model_metrics_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved metrics comparison for {config_name}")

def plot_sensitivity_specificity_scatter(all_results: Dict, config_name: str, output_dir: Path):
    """Plot sensitivity vs specificity scatter plot for all models."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model_name, data in sorted(all_results.items()):
        if not data.get('success', False):
            continue
            
        color = MODEL_COLORS.get(model_name, 'gray')
        
        ax.scatter([data['specificity']], [data['sensitivity']],
                  s=200, color=color, edgecolors='black', linewidth=2,
                  label=MODEL_NAMES.get(model_name, model_name), alpha=0.8, zorder=3)
        ax.annotate(MODEL_NAMES.get(model_name, model_name),
                   (data['specificity'], data['sensitivity']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Specificity (TNR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sensitivity (TPR)', fontsize=12, fontweight='bold')
    ax.set_title(f'{config_name}\nSensitivity vs Specificity Across Models', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.65, 0.95)
    ax.set_ylim(0.80, 1.0)
    ax.plot([0.65, 0.95], [0.65, 0.95], 'k--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    safe_name = config_name.replace(':', '_').replace('/', '_').replace(' ', '_')
    plt.savefig(output_dir / f'cross_model_sens_spec_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Saved sensitivity-specificity scatter for {config_name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Cross-Model Comparison Plots (FULLY PARALLEL)')
    parser.add_argument('--experiments_dir', type=str, default='experiments')
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--output_dir', type=str, default='experiments/evaluation/cross_model_comparisons')
    parser.add_argument('--models', type=str, nargs='+', default=None)
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-discover experiments
    exps_dir = Path(args.experiments_dir)
    exp_folders = sorted([p for p in exps_dir.glob('exp_*') if p.is_dir()])
    
    if args.models:
        exp_folders = [e for e in exp_folders if e.name in args.models]
    
    print(f" Found {len(exp_folders)} experiments")
    print(f" Using {args.workers} parallel workers")
    
    print(f"\n{'='*70}")
    print(f" PHASE 1: Parallel Model Data Loading")
    print(f"{'='*70}")
    
    worker_args = []
    for exp_folder in exp_folders:
        calib_file = exp_folder / 'calibration' / 'calibration.json'
        manifests = list((exp_folder / 'data').glob('*_manifest.json'))
        if not calib_file.exists() or not manifests:
            print(f"️  Skipping {exp_folder.name}: Missing files")
            continue
        worker_args.append((exp_folder, args.cache_dir, str(manifests[0]), str(calib_file), args.split))
    
    all_raw_data = {}
    start_load = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_load_model_data, arg) for arg in worker_args]
        for fut in as_completed(futures):
            result = fut.result()
            if 'error' in result:
                print(f"️  {result['error']}")
            else:
                model = result['model']
                print(f" Loaded {model}: {result['n_samples']} samples")
                all_raw_data[model] = {
                    'probs': result['probs'],
                    'targets': result['targets'],
                    'prevalence': result['prevalence'],
                    'n_samples': result['n_samples']
                }
    
    load_time = time.time() - start_load
    print(f"\n Data loading complete in {load_time:.2f}s")
    
    if not all_raw_data:
        print(" No valid results loaded.")
        return
    
    print(f"\n{'='*70}")
    print(f" PHASE 1.5: Aligning Models to Common Sample Size (Truncation)")
    print(f"{'='*70}")
    
    # 1. Find minimum sample count across all loaded models
    sample_counts = {name: data['n_samples'] for name, data in all_raw_data.items()}
    min_samples = min(sample_counts.values())
    
    print(f"   Sample counts: {sample_counts}")
    print(f"   ️  Truncating all models to {min_samples} samples (Random Subsample)")
    
    # 2. Randomly subsample each model to min_samples
    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    aligned_data = {}
    
    for name, data in all_raw_data.items():
        n = data['n_samples']
        if n > min_samples:
            # Generate random indices
            indices = rng.choice(n, size=min_samples, replace=False)
            indices = np.sort(indices) # Keep sorted for consistency
            
            aligned_data[name] = {
                'probs': data['probs'][indices],
                'targets': data['targets'][indices],
                'prevalence': float(np.mean(data['targets'][indices])),
                'n_samples': min_samples
            }
            print(f"    {name}: {n} → {min_samples} samples")
        else:
            # Already at min_samples
            aligned_data[name] = data
            print(f"    {name}: {n} samples (unchanged)")
    
    all_raw_data = aligned_data
    print(f"    All models now have exactly {min_samples} samples.")

    print(f"\n{'='*70}")
    print(f" PHASE 2: Parallel Metric & Curve Computation")
    print(f"{'='*70}")
    
    # Create all (model, config) combinations
    compute_args = []
    for model_name, data in all_raw_data.items():
        for config_name, costs in COST_CONFIGS.items():
            compute_args.append((
                model_name,
                config_name,
                data['probs'],
                data['targets'],
                costs['cFN'],
                costs['cFP'],
                costs['C_ref']
            ))
    
    print(f" Computing metrics for {len(compute_args)} (model, config) pairs...")
    
    all_results = {}  # {config_name: {model_name: metrics}}
    start_compute = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_compute_single_metric, arg) for arg in compute_args]
        completed = 0
        for fut in as_completed(futures):
            result = fut.result()
            
            config_name = result['config']
            model_name = result['model']
            
            if config_name not in all_results:
                all_results[config_name] = {}
            
            all_results[config_name][model_name] = result
            
            completed += 1
            if completed % 7 == 0:
                print(f"   Progress: {completed}/{len(compute_args)}")
    
    compute_time = time.time() - start_compute
    print(f"\n Metric computation complete in {compute_time:.2f}s")
    
    print(f"\n{'='*70}")
    print(f" PHASE 3: Generating Plots")
    print(f"{'='*70}")
    
    for config_name in COST_CONFIGS.keys():
        print(f"\n Processing {config_name}...")
        
        plot_cost_coverage_comparison(all_results[config_name], config_name, output_dir)
        plot_metrics_comparison(all_results[config_name], config_name, output_dir)
        plot_sensitivity_specificity_scatter(all_results[config_name], config_name, output_dir)
    
    print(f"\n{'='*70}")
    print(f" All cross-model comparisons saved to {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn', force=True)
    main()