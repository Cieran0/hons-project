import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from src.utils.cache import InferenceCache
from src.configs.selective_cost_configs import load_configs, generate_cost_configs

DERM_SENSITIVITY = 0.857
DERM_SPECIFICITY = 0.813
DERM_MISS_RATE = 1 - DERM_SENSITIVITY
DERM_FALSE_ALARM_RATE = 1 - DERM_SPECIFICITY

def get_worker_count():
    """Get optimal worker count based on platform."""
    num_cores = multiprocessing.cpu_count()
    import platform
    if platform.system() == "Windows":
        max_workers = max(1, min(28, num_cores // 2))
        print(f"Windows detected: Using {max_workers} processes")
    else:
        max_workers = num_cores
        print(f"Linux/Mac detected: Using {max_workers} processes")
    return max_workers

def apply_selective_thresholds(probs, tau1, tau2):
    decisions = np.full(len(probs), 2, dtype=int)
    decisions[probs < tau1] = 0
    decisions[probs >= tau2] = 1
    return decisions

def calculate_selective_metrics(targets, probs, tau1, tau2):
    decisions = apply_selective_thresholds(probs, tau1, tau2)
    n_neg = int(np.sum(targets == 0))
    n_pos = int(np.sum(targets == 1))
    
    tn = int(np.sum((decisions == 0) & (targets == 0)))
    rn_neg = int(np.sum((decisions == 2) & (targets == 0)))
    fp = int(np.sum((decisions == 1) & (targets == 0)))
    fn = int(np.sum((decisions == 0) & (targets == 1)))
    rn_pos = int(np.sum((decisions == 2) & (targets == 1)))
    tp = int(np.sum((decisions == 1) & (targets == 1)))
    
    tnr = tn / n_neg if n_neg > 0 else 0
    rnr = rn_neg / n_neg if n_neg > 0 else 0
    fpr = fp / n_neg if n_neg > 0 else 0
    fnr = fn / n_pos if n_pos > 0 else 0
    rpr = rn_pos / n_pos if n_pos > 0 else 0
    tpr = tp / n_pos if n_pos > 0 else 0
    
    coverage = 1 - (rn_neg + rn_pos) / len(targets)
    non_rejected = decisions != 2
    n_non_rejected = np.sum(non_rejected)
    
    if n_non_rejected > 0:
        accuracy = np.sum((decisions[non_rejected] == targets[non_rejected])) / n_non_rejected
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    else:
        accuracy = 0
        ppv = 0
        npv = 0
    
    return {
        'tau1': float(tau1),
        'tau2': float(tau2),
        'tn': tn, 'rn_neg': rn_neg, 'fp': fp,
        'fn': fn, 'rn_pos': rn_pos, 'tp': tp,
        'tnr': float(tnr), 'rnr': float(rnr), 'fpr': float(fpr),
        'fnr': float(fnr), 'rpr': float(rpr), 'tpr': float(tpr),
        'coverage': float(coverage), 'accuracy': float(accuracy),
        'ppv': float(ppv), 'npv': float(npv),
        'total_samples': len(targets),
        'rejected_count': int(rn_neg + rn_pos),
        'n_pos': n_pos, 'n_neg': n_neg,
    }

def calculate_true_cost_average(targets, decisions, cost_fn, cost_fp, cRP, cRN, cost_tn=0, cost_tp=0):
    n = len(targets)
    total_cost = 0
    for i in range(n):
        y = int(targets[i])
        d = int(decisions[i])
        if d == 2:
            total_cost += cRP if y == 1 else cRN
        elif d == y:
            total_cost += cost_tp if y == 1 else cost_tn
        elif y == 1:
            total_cost += cost_fn
        else:
            total_cost += cost_fp
    return total_cost / n

def calculate_expected_cost_from_rates(metrics, cost_fn, cost_fp, cRP, cRN, prevalence):
    pi = prevalence
    cost_errors = (pi * cost_fn * metrics['fnr'] + 
                   (1 - pi) * cost_fp * metrics['fpr'])
    cost_rejections = (pi * cRP * metrics['rpr'] +
                       (1 - pi) * cRN * metrics['rnr'])
    return cost_errors + cost_rejections

def derive_rejection_costs(cost_fn, cost_fp, C_ref):
    cRP = DERM_MISS_RATE * cost_fn + C_ref
    cRN = DERM_FALSE_ALARM_RATE * cost_fp + C_ref
    return cRP, cRN

def find_optimal_thresholds(probs, targets, cost_fn, cost_fp, C_ref, prevalence):
    cRP, cRN = derive_rejection_costs(cost_fn, cost_fp, C_ref)
    
    if cRP >= cost_fn or cRN >= cost_fp:
        return 0.0, 1.0, None, None, None
    
    tau1 = cRN / (cRN + cost_fn - cRP)
    tau2 = (cRN - cost_fp) / (cRN - cost_fp - cRP)
    
    if tau1 >= tau2:
        return 0.0, 1.0, None, None, None
    
    tau1 = max(0.0, min(1.0, tau1))
    tau2 = max(0.0, min(1.0, tau2))
    
    metrics = calculate_selective_metrics(targets, probs, tau1, tau2)
    
    expected_cost = calculate_expected_cost_from_rates(
        metrics, cost_fn, cost_fp, cRP, cRN, prevalence
    )
    
    decisions = apply_selective_thresholds(probs, tau1, tau2)
    true_avg_cost = calculate_true_cost_average(
        targets, decisions, cost_fn, cost_fp, cRP, cRN
    )
    
    return tau1, tau2, true_avg_cost, expected_cost, metrics

def process_config_worker(args):
    config, probs, targets, prevalence = args
    
    cost_fn = config['cost_fn']
    cost_fp = config['cost_fp']
    C_ref = config.get('C_ref', config.get('cost_r', 0))
    cR_ratio = config.get('cR_ratio', C_ref / min(cost_fn, cost_fp))
    config_name = config.get('cost_config', f"FP:FN={config['fp_fn_name']}")
    
    tau1, tau2, true_avg_cost, expected_cost, metrics = find_optimal_thresholds(
        probs, targets,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        C_ref=C_ref,
        prevalence=prevalence
    )
    
    if metrics is None:
        standard_preds = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(targets, standard_preds).ravel()
        true_avg_cost = (fn * cost_fn + fp * cost_fp) / len(targets)
        expected_cost = true_avg_cost
        metrics = {
            'tau1': 0.0,
            'tau2': 1.0,
            'coverage': 1.0,
            'reject_rate': 0.0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'rpr': 0.0,
            'rnr': 0.0,
            'total_samples': len(targets),
            'rejected_count': 0,
        }
    
    true_total_cost = true_avg_cost * len(targets)
    
    metrics['cost_config'] = config_name
    metrics['cR_ratio'] = cR_ratio
    metrics['cost_fn'] = cost_fn
    metrics['cost_fp'] = cost_fp
    metrics['C_ref'] = C_ref
    metrics['true_avg_cost'] = true_avg_cost
    metrics['true_total_cost'] = true_total_cost
    metrics['expected_cost'] = expected_cost
    metrics['reject_rate'] = 1 - metrics['coverage']
    metrics['config_id'] = config.get('id', 0)
    
    cRP, cRN = derive_rejection_costs(cost_fn, cost_fp, C_ref)
    metrics['cRP'] = cRP
    metrics['cRN'] = cRN
    
    return metrics


def main(args):
    print("Selective Classification Experiment")
    print("="*70)
    
    max_workers = get_worker_count()
    num_cores = multiprocessing.cpu_count()
    
    print(f"Detected {num_cores} CPU cores")
    print(f"   Using {max_workers} processes for parallel processing")
    
    cache = InferenceCache(cache_dir=args.cache_dir)
    
    with open(args.calibration) as f:
        calib = json.load(f)
    temperature = calib['temperature']
    print(f"\nTemperature: {temperature:.4f}")
    
    probs, targets, image_ids, meta = cache.load(
        args.checkpoint, args.manifest, args.split, temperature=temperature
    )
    if probs is None:
        print("No cached inference found! Run run_inference.py first.")
        return
    print(f"Loaded {len(probs)} predictions from cache")
    
    standard_preds = (probs >= 0.5).astype(int)
    standard_tn, standard_fp, standard_fn, standard_tp = confusion_matrix(targets, standard_preds).ravel()
    standard_tpr = standard_tp / (standard_tp + standard_fn) if (standard_tp + standard_fn) > 0 else 0
    standard_fpr = standard_fp / (standard_fp + standard_tn) if (standard_fp + standard_tn) > 0 else 0
    standard_acc = (standard_tp + standard_tn) / len(targets)
    standard_true_cost = calculate_true_cost_average(
        targets, standard_preds, cost_fn=1, cost_fp=1, cRP=0, cRN=0
    )
    print(f"\nStandard Classification (threshold=0.5):")
    print(f"   TRUE COST (avg/sample): {standard_true_cost:.4f}")
    
    prevalence = np.mean(targets)
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    print(f"\nTest set prevalence: {prevalence:.2%}")
    
    print(f"\n{'='*70}")
    print(f"Loading Cost Configurations (FN >= FP only)")
    print(f"{'='*70}")
    if args.config_file and Path(args.config_file).exists():
        print(f"Loading configs from {args.config_file}")
        config_data = load_configs(args.config_file)
        configs = config_data if isinstance(config_data, list) else config_data.get('configurations', [])
        configs = [c for c in configs if c['cost_fp'] <= c['cost_fn']]
        print(f"    Loaded {len(configs)} configurations from file (filtered)")
    else:
        print(f"Generating configurations...")
        configs = generate_cost_configs(max_power=20, fn_only=True)
        print(f"   Generated {len(configs)} configurations (FN >= FP only)")
    
    print(f"\n{'='*70}")
    print(f"Running Selective Classification Experiments (Parallel)")
    print(f"{'='*70}\n")
    results = []
    total_configs = len(configs)
    
    worker_args = [(config, probs, targets, prevalence) for config in configs]
    
    print(f"⚡ Starting parallel processing with {max_workers} processes...")
    import time
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_config_worker, arg) for arg in worker_args]
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_configs - completed) / rate if rate > 0 else 0
                print(f"   Progress: {completed}/{total_configs} ({completed/total_configs*100:.1f}%) - ETA: {eta:.1f}s")
    
    elapsed_time = time.time() - start_time
    print(f"\nParallel processing complete!")
    print(f"   Total time: {elapsed_time:.2f}s")
    print(f"   Average per config: {elapsed_time/total_configs*1000:.2f}ms")
    
    results_dir = Path(args.output_dir) / 'selective_classification'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = results_dir / f'selective_results_{args.split}_expanded.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Saved detailed results to {results_path}")
    
    summary = {
        'split': args.split,
        'total_samples': len(probs),
        'prevalence': float(prevalence),
        'n_pos': n_pos,
        'n_neg': n_neg,
        'temperature': float(temperature),
        'config_file': args.config_file if args.config_file else 'generated',
        'total_configurations': len(results),
        'results_file': str(results_path),
        'processing': {
            'num_cores': num_cores,
            'max_workers': max_workers,
            'executor_type': 'ProcessPoolExecutor',
            'total_time_seconds': elapsed_time,
            'avg_time_per_config_ms': elapsed_time/total_configs*1000,
        },
        'standard_classification': {
            'accuracy': float(standard_acc),
            'tpr': float(standard_tpr),
            'fpr': float(standard_fpr),
            'true_avg_cost': float(standard_true_cost),
        },
        'paper_compliance': {
            'threshold_method': 'closed_form_eq_4_5',
            'rejection_cost_method': 'expert_adjusted_eq_2_3',
            'dermatologist_sensitivity': DERM_SENSITIVITY,
            'dermatologist_specificity': DERM_SPECIFICITY,
        },
        'clinical_validity': {
            'fn_geq_fp_only': True,
        }
    }
    summary_path = results_dir / f'selective_summary_{args.split}_expanded.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    print(f"\n{'='*70}")
    print(f"TOP 10 CONFIGURATIONS (by TRUE Average Cost)")
    print(f"{'='*70}")
    results_df_sorted = results_df.sort_values('true_avg_cost').head(10)
    for idx, row in results_df_sorted.iterrows():
        print(f"\n{row['cost_config']}, C_ref={row['C_ref']:.1f}")
        print(f"   TRUE Cost (avg/sample): {row['true_avg_cost']:.4f}")
        print(f"   Expected Cost (deployment): {row['expected_cost']:.4f}")
        print(f"   Coverage: {row['coverage']:.2%}")
        print(f"   τ1: {row['tau1']:.4f}, τ2: {row['tau2']:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Selective Classification Experiment Complete!")
    print(f"{'='*70}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selective Classification with Reject Option')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--calibration', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to cost config JSON file')
    args = parser.parse_args()
    main(args)