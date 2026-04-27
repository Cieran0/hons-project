import sys
import argparse
import json
import multiprocessing
import platform
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.cache import InferenceCache
from src.analysis.core_metrics import (
    derive_rejection_costs, find_optimal_thresholds, 
    compute_confusion_matrix_3x2, compute_rates_from_confusion,
    generate_cost_coverage_curve, calculate_sample_uncertainty
)
from src.visualization.plotters import (
    plot_3x2_confusion, plot_cost_coverage_curves, 
    plot_cost_vs_cref_sweep, plot_3d_surface, plot_2d_contour
)

from src.configs.selective_cost_configs import generate_cost_configs, generate_fp_fn_ratios
from src.configs.real_life_configs import generate_real_life_configs


def _confusion_worker(args):
    config, probs, targets = args
    if 'tau1' not in config or config.get('tau1') == 0.5:
         t1, t2 = find_optimal_thresholds(config['cost_fn'], config['cost_fp'], config.get('C_ref', 0))
         config['tau1'], config['tau2'] = t1, t2
    
    cm = compute_confusion_matrix_3x2(targets, probs, config['tau1'], config['tau2'])
    rates = compute_rates_from_confusion(cm)
    cov = (cm['TN'] + cm['TP'] + cm['FP'] + cm['FN']) / (cm['n_neg'] + cm['n_pos'])
    config['coverage'] = cov
    
    safe_cm = {k: int(v) for k, v in cm.items()}
    
    return {'config': config, 'cm': safe_cm, 'rates': rates}

def _cost_coverage_worker(args):
    config, probs, targets = args
    C_ref = config.get('C_ref', 0)
    df = generate_cost_coverage_curve(probs, targets, config['cost_fn'], config['cost_fp'], C_ref, use_baseline_norm=True)
    name = config.get('fp_fn_name', config.get('cost_config', 'Unknown'))
    return {'name': name, 'df': df}

def _get_coverage(probs, cost_fn, cost_fp, c_ref):
    """Quick coverage calculation for endpoint search."""
    tau1, tau2 = find_optimal_thresholds(cost_fn, cost_fp, c_ref)
    
    decisions = np.full(len(probs), 2, dtype=int)
    if tau1 >= tau2:
        decisions[probs < 0.5] = 0
        decisions[probs >= 0.5] = 1
    else:
        decisions[probs < tau1] = 0
        decisions[probs >= tau2] = 1
    
    return np.sum(decisions != 2) / len(probs)

def _find_100_coverage_endpoint(probs, cost_fn, cost_fp):
    """Find exact C_ref where coverage = 100% using adaptive search."""
    c_ref = 0.0
    step = max(cost_fn, cost_fp) * 1.0
    
    # Phase 1: Go up until we hit 100%
    for _ in range(1000):
        coverage = _get_coverage(probs, cost_fn, cost_fp, c_ref)
        if coverage >= 0.995:
            break
        c_ref += step
    
    # Phase 2-6: Refine with decreasing step sizes
    for step_mult in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        step = max(cost_fn, cost_fp) * step_mult
        
        # Go back until < 100%
        for _ in range(100):
            coverage = _get_coverage(probs, cost_fn, cost_fp, c_ref)
            if coverage < 0.995:
                break
            c_ref -= step
        
        # Go forward until >= 100%
        for _ in range(100):
            coverage = _get_coverage(probs, cost_fn, cost_fp, c_ref)
            if coverage >= 0.995:
                break
            c_ref += step
    
    return c_ref

def _find_0_coverage_endpoint(probs, cost_fn, cost_fp):
    """Find exact C_ref where coverage = 0% using adaptive search."""
    c_ref = 0.0
    step = max(cost_fn, cost_fp) * 1.0
    
    # Phase 1: Go down until we hit 0%
    for _ in range(1000):
        coverage = _get_coverage(probs, cost_fn, cost_fp, c_ref)
        if coverage <= 0.005:
            break
        c_ref -= step
    
    # Phase 2-6: Refine with decreasing step sizes
    for step_mult in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        step = max(cost_fn, cost_fp) * step_mult
        
        # Go back until > 0%
        for _ in range(100):
            coverage = _get_coverage(probs, cost_fn, cost_fp, c_ref)
            if coverage > 0.005:
                break
            c_ref += step
        
        # Go forward until <= 0%
        for _ in range(100):
            coverage = _get_coverage(probs, cost_fn, cost_fp, c_ref)
            if coverage <= 0.005:
                break
            c_ref -= step
    
    return c_ref

def _cref_sweep_worker(args):
    """
    Find exact 0% and 100% coverage endpoints.
    Generates three sweep types per config:
    Full transition: 0% coverage - 100% coverage (includes negative C_ref)
    C_ref=0 start: C_ref=0 - 100% coverage (no negative costs, more interpretable)
    Positive only: C_ref=0 - C_ref@100%
    """
    config, probs, targets = args
    cost_fn = config['cost_fn']
    cost_fp = config['cost_fp']
    config_name = config.get('fn_fp_name', f"FN:FP={cost_fn}:{cost_fp}")
    
    c_ref_100 = _find_100_coverage_endpoint(probs, cost_fn, cost_fp)
    c_ref_0 = _find_0_coverage_endpoint(probs, cost_fn, cost_fp)
    
    print(f"   Coverage range: C_ref [{c_ref_0:.4f}, {c_ref_100:.4f}]")
    
    c_ref_values_full = np.linspace(c_ref_0, c_ref_100, 1000)
    
    c_ref_values_zero_start = np.linspace(0.0, c_ref_100, 1000)
    
    results_full = []
    results_zero_start = []
    
    for c_ref in c_ref_values_full:
        tau1, tau2 = find_optimal_thresholds(cost_fn, cost_fp, c_ref)
        
        decisions = np.full(len(probs), 2, dtype=int)
        if tau1 >= tau2:
            decisions[probs < 0.5] = 0
            decisions[probs >= 0.5] = 1
        else:
            decisions[probs < tau1] = 0
            decisions[probs >= tau2] = 1
        
        cls_mask = decisions != 2
        n_cls = np.sum(cls_mask)
        
        if n_cls == 0:
            continue
        
        fn = np.sum((decisions == 0) & (targets == 1))
        fp = np.sum((decisions == 1) & (targets == 0))
        cost_removed = (fn * cost_fn + fp * cost_fp) / n_cls
        cov = n_cls / len(probs)
        
        results_full.append({
            'config_name': config_name,
            'c_ref': c_ref,
            'avg_true_cost_removed': cost_removed,
            'coverage': cov,
            'c_ref_at_0': c_ref_0,
            'c_ref_at_100': c_ref_100,
            'sweep_type': 'full_transition',
        })
    
    for c_ref in c_ref_values_zero_start:
        tau1, tau2 = find_optimal_thresholds(cost_fn, cost_fp, c_ref)
        
        decisions = np.full(len(probs), 2, dtype=int)
        if tau1 >= tau2:
            decisions[probs < 0.5] = 0
            decisions[probs >= 0.5] = 1
        else:
            decisions[probs < tau1] = 0
            decisions[probs >= tau2] = 1
        
        cls_mask = decisions != 2
        n_cls = np.sum(cls_mask)
        
        if n_cls == 0:
            continue
        
        fn = np.sum((decisions == 0) & (targets == 1))
        fp = np.sum((decisions == 1) & (targets == 0))
        cost_removed = (fn * cost_fn + fp * cost_fp) / n_cls
        cov = n_cls / len(probs)
        
        results_zero_start.append({
            'config_name': config_name,
            'c_ref': c_ref,
            'avg_true_cost_removed': cost_removed,
            'coverage': cov,
            'c_ref_at_0': 0.0,  # Start at 0
            'c_ref_at_100': c_ref_100,
            'sweep_type': 'zero_start',
        })
    
    df_full = pd.DataFrame(results_full)
    df_zero_start = pd.DataFrame(results_zero_start)
    
    return df_full, df_zero_start, c_ref_0, c_ref_100

def _surface_worker(args):
    probs, targets, cfn_ratios, cref_ratios = args
    cost_fp = 1.0
    
    Z_cost_relative = np.zeros((len(cref_ratios), len(cfn_ratios)))
    Z_cov = np.zeros((len(cref_ratios), len(cfn_ratios)))
    
    # Baseline counts (no selective classification, threshold 0.5)
    baseline_decisions = (probs >= 0.5).astype(int)
    baseline_fn = np.sum((baseline_decisions == 0) & (targets == 1))
    baseline_fp = np.sum((baseline_decisions == 1) & (targets == 0))
    
    for i, cfn_r in enumerate(cfn_ratios):
        cost_fn = cost_fp * cfn_r
        # Baseline cost scales with the FN:FP ratio
        baseline_cost = baseline_fn * cost_fn + baseline_fp * cost_fp
        
        for j, cref_r in enumerate(cref_ratios):
            C_ref = cost_fp * cref_r
            tau1, tau2 = find_optimal_thresholds(cost_fn, cost_fp, C_ref)
            
            decisions = np.full(len(probs), 2, dtype=int)
            if tau1 >= tau2:
                decisions[probs < 0.5] = 0
                decisions[probs >= 0.5] = 1
            else:
                decisions[probs < tau1] = 0
                decisions[probs >= tau2] = 1
            
            cls_mask = decisions != 2
            n_cls = np.sum(cls_mask)
            Z_cov[j, i] = n_cls / len(probs)
            
            if n_cls > 0 and baseline_cost > 0:
                fn = np.sum((decisions == 0) & (targets == 1))
                fp = np.sum((decisions == 1) & (targets == 0))
                selective_cost = fn * cost_fn + fp * cost_fp
                
                # Relative cost as fraction of baseline
                Z_cost_relative[j, i] = selective_cost / baseline_cost
            else:
                Z_cost_relative[j, i] = np.nan
                
    return Z_cost_relative, Z_cov, baseline_fn, baseline_fp

def get_worker_count():
    num_cores = multiprocessing.cpu_count()
    if platform.system() == "Windows":
        return max(1, min(28, num_cores // 2))
    return num_cores

def run_confusion_analysis(args, probs, targets, prevalence):
    print(f"\nRunning Confusion Matrix Analysis: Strategy '{args.strategy}'")
    configs = []
    
    if args.strategy == 'extremes':
        results_path = Path(args.output_dir) / 'selective_classification' / f'selective_results_{args.split}_expanded.csv'
        if not results_path.exists():
            print("Results CSV not found for 'extremes'.")
            return
        df = pd.read_csv(results_path)
        df = df[df['cost_fp'] <= df['cost_fn']] 
        
        def add_row_metric(row, metric_name, metric_val):
            configs.append({
                'config_name': row['cost_config'], 
                'cost_fn': row['cost_fn'], 
                'cost_fp': row['cost_fp'], 
                'C_ref': row.get('C_ref', row.get('cost_r', 0)),
                'tau1': row['tau1'], 
                'tau2': row['tau2'], 
                'coverage': row['coverage'],
            })

        if 'fpr' in df.columns:
            idx = df['fpr'].idxmin()
            add_row_metric(df.loc[idx], 'Lowest FPR', df.loc[idx]['fpr'])
        if 'fnr' in df.columns:
            idx = df['fnr'].idxmin()
            add_row_metric(df.loc[idx], 'Lowest FNR', df.loc[idx]['fnr'])
        
        min_cov = getattr(args, 'min_coverage', 0.50)
        valid_df = df[df['coverage'] >= min_cov]
        if not valid_df.empty:
            sort_col = 'expected_cost' if 'expected_cost' in valid_df.columns else 'true_avg_cost'
            if sort_col in valid_df.columns:
                idx = valid_df[sort_col].idxmin()
                add_row_metric(valid_df.loc[idx], 'Lowest Cost', valid_df.loc[idx][sort_col])

    elif args.strategy == 'real_life':
        raw_configs = generate_real_life_configs(fn_tier_names=args.tiers, normalize_by_fn=False)
        for c in raw_configs:
            c['config_name'] = c.get('fp_fn_name', c['cost_config'])
            c['C_ref'] = c.get('cost_r', 227.0)
            t1, t2 = find_optimal_thresholds(c['cost_fn'], c['cost_fp'], c['C_ref'])
            c['tau1'], c['tau2'] = t1, t2
        configs = raw_configs
        
    elif args.strategy == 'all_ratios':
        ratios = generate_fp_fn_ratios(max_power=getattr(args, 'max_power', 12), fn_only=True)
        for r in ratios:
            configs.append({
                'config_name': f"FP:FN={r['name']}",
                'cost_fn': r['cost_fn'], 
                'cost_fp': r['cost_fp'],
                'C_ref': 0, 
                'tau1': 0.5, 
                'tau2': 0.5
            })

    if not configs:
        print("No configurations generated.")
        return

    print(f"Selected {len(configs)} configurations.")
    worker_args = [(c, probs, targets) for c in configs]
    results = {}
    max_workers = get_worker_count()
    
    print(f"Processing with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_confusion_worker, arg) for arg in worker_args]
        for fut in as_completed(futures):
            res = fut.result()
            name = res['config']['config_name']
            results[name] = res
            print(f"   Completed {name}")

    out_dir = Path(args.output_dir) / 'visualization' / f'confusion_{args.strategy}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for name, res in results.items():
        safe_name = name.replace(":", "_").replace("/", "_")
        plot_3x2_confusion(res['cm'], name, res['config']['coverage'], 
                           out_dir / f'{safe_name}_norm.png', mode='normalized')
        plot_3x2_confusion(res['cm'], name, res['config']['coverage'], 
                           out_dir / f'{safe_name}_counts.png', mode='counts')
    print(f"Saved matrices to {out_dir}")

def run_cost_coverage_analysis(args, probs, targets):
    print(f"\nRunning Cost-Coverage Analysis: Strategy '{args.strategy}'")
    configs = []
    
    if args.strategy == 'top_n':
        results_path = Path(args.output_dir) / 'selective_classification' / f'selective_results_{args.split}_expanded.csv'
        if not results_path.exists():
            print(f"Results CSV not found: {results_path}")
            return
        
        print(f"Loading results from: {results_path}")
        df = pd.read_csv(results_path)
        print(f"  Loaded {len(df)} rows from CSV")
        
        df = df[df['cost_fp'] <= df['cost_fn']]
        print(f"  After FN>=FP filter: {len(df)} rows")
        
        sort_col = None
        for col in ['true_avg_cost_norm', 'avg_true_cost_norm', 'true_avg_cost', 'avg_true_cost']:
            if col in df.columns:
                sort_col = col
                break
        
        if sort_col:
            top_df = df.nsmallest(getattr(args, 'top_n', 20), sort_col)
        else:
            top_df = df.head(20)
        
        print(f"  Top {len(top_df)} configs selected")
        
        all_cfgs = generate_cost_configs(max_power=31, fn_only=True)
        cfg_map = {c['cost_config']: c for c in all_cfgs}
        
        matched = 0
        for _, row in top_df.iterrows():
            config_name = None
            for col_name in ['cost_config', 'config_name', 'fp_fn_name', 'name']:
                if col_name in row and row[col_name] in cfg_map:
                    config_name = row[col_name]
                    break
            
            if config_name:
                c = cfg_map[config_name].copy()
                c['C_ref'] = row.get('C_ref', row.get('cost_r', 0))
                configs.append(c)
                matched += 1
            else:
                if 'cost_fn' in row and 'cost_fp' in row:
                    configs.append({
                        'cost_config': row.get('cost_config', f"FN:FP={int(row['cost_fn'])}:{int(row['cost_fp'])}"),
                        'cost_fn': row['cost_fn'],
                        'cost_fp': row['cost_fp'],
                        'C_ref': row.get('C_ref', row.get('cost_r', 0)),
                        'fp_fn_name': row.get('cost_config', f"FN:FP={int(row['cost_fn'])}:{int(row['cost_fp'])}")
                    })
                    matched += 1
        
        print(f"  Matched {matched}/{len(top_df)} configs")
                
    elif args.strategy == 'real_life':
        configs = generate_real_life_configs(fn_tier_names=args.tiers, normalize_by_fn=False)
        for c in configs: 
            c['C_ref'] = c.get('cost_r', 227.0)

    if not configs:
        print("No configurations generated.")
        return

    worker_args = [(c, probs, targets) for c in configs]
    results_dict = {}
    max_workers = get_worker_count()
    
    print(f"Processing with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_cost_coverage_worker, arg) for arg in worker_args]
        for fut in as_completed(futures):
            res = fut.result()
            results_dict[res['name']] = res['df']
            print(f"   Curve generated for {res['name']}")

    out_dir = Path(args.output_dir) / 'visualization' / f'cost_cov_{args.strategy}'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_cost_coverage_curves(
        results_dict, 
        out_dir / 'avg_removed_norm.png',
        'avg_true_cost_removed',
        f'Average True Cost (Removed) - {args.strategy.upper()}',
        normalize=True
    )
    plot_cost_coverage_curves(
        results_dict, 
        out_dir / 'total_removed_norm.png',
        'total_true_cost_removed',
        f'Total True Cost (Removed) - {args.strategy.upper()}',
        normalize=True
    )
    print(f"Saved curves to {out_dir}")

def _plot_cref_sweep_line(df_list, output_dir, split_name, n_samples, sweep_type='full_transition'):
    """
    Plot individual line sweeps.
    sweep_type: 'full_transition' (0%→100% coverage) or 'zero_start' (C_ref=0→100%)
    """
    full_df = pd.concat(df_list, ignore_index=True)
    configs = full_df['config_name'].unique()
    
    for cfg in configs:
        df = full_df[full_df['config_name'] == cfg].sort_values('c_ref')
        
        # Get endpoints from data
        c_ref_0 = df['c_ref_at_0'].iloc[0] if 'c_ref_at_0' in df.columns else df['c_ref'].min()
        c_ref_100 = df['c_ref_at_100'].iloc[0] if 'c_ref_at_100' in df.columns else df['c_ref'].max()
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color_cost = '#E63946'
        ax1.plot(df['c_ref'], df['avg_true_cost_removed'], 
                 linewidth=2.5, color=color_cost, label='Avg Cost (Removed)')
        ax1.set_xlabel('Referral Cost (C_ref)', fontsize=12)
        ax1.set_ylabel('Avg True Cost of Samples Classified', fontsize=12, color=color_cost)
        ax1.tick_params(axis='y', labelcolor=color_cost)
        
        ax2 = ax1.twinx()
        color_cov = '#2E86AB'
        ax2.plot(df['c_ref'], df['coverage'], 
                 linewidth=1.5, color=color_cov, linestyle='--', label='Coverage')
        ax2.set_ylabel('Coverage', fontsize=12, color=color_cov)
        ax2.tick_params(axis='y', labelcolor=color_cov)
        ax2.set_ylim(0, 1.05)
        
        # X-axis EXACTLY from endpoints
        ax1.set_xlim(c_ref_0, c_ref_100)
        
        # Mark endpoints
        ax1.axvline(x=c_ref_0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax1.axvline(x=c_ref_100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        if sweep_type == 'zero_start':
            ax1.text(c_ref_0, ax1.get_ylim()[1]*0.95, 'C_ref = 0', 
                     rotation=90, va='top', ha='right', fontsize=9, color='gray')
        else:
            ax1.text(c_ref_0, ax1.get_ylim()[1]*0.95, '0% Coverage', 
                     rotation=90, va='top', ha='right', fontsize=9, color='gray')
        
        ax1.text(c_ref_100, ax1.get_ylim()[1]*0.95, '100% Coverage', 
                 rotation=90, va='top', ha='left', fontsize=9, color='gray')

        title_suffix = '(C_ref ≥ 0)' if sweep_type == 'zero_start' else '(Full Transition)'
        ax1.set_title(f'Referral Cost Sensitivity \n{cfg}', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        safe_name = cfg.replace(":", "_").replace("/", "_")
        plt.tight_layout()
        plt.savefig(output_dir / f'sweep_line_{sweep_type}_{safe_name}_{split_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Combined plot
    fig, ax = plt.subplots(figsize=(14, 9))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs)))
    
    for idx, cfg in enumerate(sorted(configs)):
        df = full_df[full_df['config_name'] == cfg].sort_values('c_ref')
        ax.plot(df['c_ref'], df['avg_true_cost_removed'], 
                label=cfg, linewidth=2, color=colors[idx], alpha=0.8)
                
    ax.set_xlabel('Referral Cost (C_ref)', fontsize=12)
    ax.set_ylabel('Avg True Cost (Removed)', fontsize=12)
    title_suffix = '(C_ref ≥ 0)' if sweep_type == 'zero_start' else '(Full Transition)'
    ax.set_title(f'Combined Sensitivity: Avg True Cost (Removed) vs C_ref {title_suffix}\n{split_name} Set', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Configurations', loc='upper right', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'sweep_combined_{sweep_type}_{split_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined line plot ({sweep_type})")

def run_sensitivity_and_surfaces(args, probs, targets):
    print(f"\nRunning High-Res Sensitivity/Surface Analysis")
    
    n_samples = len(probs)
    max_workers = get_worker_count()
    
    ratios = []
    base_fp = 1
    effective_max_power = min(getattr(args, 'max_power', 7), 7)
    for power in range(0, effective_max_power + 1):
        fn_mult = 2 ** power
        ratios.append({
            'cost_fn': fn_mult * base_fp,
            'cost_fp': base_fp,
            'fn_fp_name': f'FN:FP={fn_mult}:1'
        })
        
    print(f"Generated {len(ratios)} configs")
    
    output_dir = Path(args.output_dir) / 'visualization' / 'cref_sensitivity_highres'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if getattr(args, 'do_sensitivity', False):
        print(f"\nRunning High-Res Line Sweeps...")
        worker_args = [(cfg, probs, targets) for cfg in ratios]
        
        dfs_full = []
        dfs_zero_start = []
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_cref_sweep_worker, arg) for arg in worker_args]
            for fut in as_completed(futures):
                df_full, df_zero_start, c_ref_0, c_ref_100 = fut.result()
                dfs_full.append(df_full)
                dfs_zero_start.append(df_zero_start)
                
        elapsed = time.time() - start_time
        print(f"Line sweeps complete in {elapsed:.2f}s")
        
        # Plot full transition sweeps
        print(f"  Plotting full transition sweeps...")
        _plot_cref_sweep_line(dfs_full, output_dir, args.split, n_samples, sweep_type='full_transition')
        pd.concat(dfs_full).to_csv(output_dir / f'sweep_data_full_{args.split}.csv', index=False)
        
        # Plot C_ref=0 start sweeps
        print(f"  Plotting C_ref=0 start sweeps...")
        _plot_cref_sweep_line(dfs_zero_start, output_dir, args.split, n_samples, sweep_type='zero_start')
        pd.concat(dfs_zero_start).to_csv(output_dir / f'sweep_data_zero_start_{args.split}.csv', index=False)
    
    if getattr(args, 'do_surface', False):
        print(f"\nGenerating Cost Reduction Heatmap...")
        
        cfn_ratios = np.linspace(1, 16, 100)
        cref_ratios = np.linspace(0, 0.6, 100)
        
        surface_args = (probs, targets, cfn_ratios, cref_ratios)
        Z_cost_relative, Z_cov, baseline_fn, baseline_fp = _surface_worker(surface_args)
        
        X, Y = np.meshgrid(cfn_ratios, cref_ratios)
        
        # Convert relative cost to percentage reduction
        cost_reduction_pct = (1.0 - Z_cost_relative) * 100.0
        
        # Cap colourbar to actual data range (ignore NaNs)
        valid_vals = cost_reduction_pct[~np.isnan(cost_reduction_pct)]
        vmin_pct = np.min(valid_vals)
        vmax_pct = np.max(valid_vals)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(X, Y, cost_reduction_pct, cmap='RdYlGn', shading='auto', vmin=vmin_pct, vmax=vmax_pct)
        ax.set_xlabel('Cost of False Negative', fontsize=11)
        ax.set_ylabel('Cost of Referral (C_ref)', fontsize=11)
        ax.set_title(f'Cost Reduction vs Baseline - Cost of FP = 1', fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cost Reduction (%)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'heatmap_cost_reduction_pct_{args.split}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        surf_df = pd.DataFrame({
            'cfn_ratio': X.flatten(),
            'cref_ratio': Y.flatten(),
            'relative_cost': Z_cost_relative.flatten(),
            'cost_reduction_pct': cost_reduction_pct.flatten()
        })
        surf_df.to_csv(output_dir / f'surface_data_{args.split}.csv', index=False)
        
        print(f"\nBaseline: FN={baseline_fn}, FP={baseline_fp}")
        print(f"Cost reduction range: {vmin_pct:.2f}% to {vmax_pct:.2f}%")
    
    print(f"\nComplete! Output saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Unified Analysis Driver')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--calibration', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['confusion', 'cost_coverage', 'sensitivity_surface'])
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['extremes', 'real_life', 'all_ratios', 'top_n'])
    
    parser.add_argument('--tiers', type=str, nargs='+', default=None)
    parser.add_argument('--top-n', type=int, default=20, dest='top_n')
    parser.add_argument('--max-power', type=int, default=7, dest='max_power')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--min-coverage', type=float, default=0.50, dest='min_coverage')
    parser.add_argument('--do-sensitivity', action='store_true', dest='do_sensitivity')
    parser.add_argument('--do-surface', action='store_true', dest='do_surface')
    
    args = parser.parse_args()
    
    cache = InferenceCache(cache_dir=args.cache_dir)
    try:
        with open(args.calibration) as f:
            calib = json.load(f)
    except FileNotFoundError:
        print(f"Calibration file not found: {args.calibration}")
        return
    
    probs, targets, _, _ = cache.load(args.checkpoint, args.manifest, args.split, temperature=calib.get('temperature', 1.0))
    if probs is None:
        print("Failed to load data.")
        return
        
    print(f"Loaded {len(probs)} samples. Prevalence: {np.mean(targets):.2%}")
    
    if args.mode == 'confusion':
        run_confusion_analysis(args, probs, targets, np.mean(targets))
    elif args.mode == 'cost_coverage':
        run_cost_coverage_analysis(args, probs, targets)
    elif args.mode == 'sensitivity_surface':
        run_sensitivity_and_surfaces(args, probs, targets)

if __name__ == '__main__':
    main()