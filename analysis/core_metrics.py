import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import pandas as pd


DERM_SENSITIVITY = 0.857
DERM_SPECIFICITY = 0.813

DERM_MISS_RATE = 1 - DERM_SENSITIVITY
DERM_FALSE_ALARM_RATE = 1 - DERM_SPECIFICITY

def derive_rejection_costs(cost_fn: float, cost_fp: float, C_ref: float) -> Tuple[float, float]:
    cRP = DERM_MISS_RATE * cost_fn + C_ref
    cRN = DERM_FALSE_ALARM_RATE * cost_fp + C_ref
    return cRP, cRN

def find_optimal_thresholds(cost_fn: float, cost_fp: float, C_ref: float) -> Tuple[float, float]:
    cRP, cRN = derive_rejection_costs(cost_fn, cost_fp, C_ref)
    
    # Reject only if it reduces expected cost
    if cRP >= cost_fn or cRN >= cost_fp:
        return 0.5, 0.5
    
    denom1 = cRN + cost_fn - cRP
    denom2 = cRN - cost_fp - cRP
    
    if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
        return 0.5, 0.5
        
    tau1 = cRN / denom1
    tau2 = (cRN - cost_fp) / denom2
    
    if tau1 >= tau2:
        return 0.5, 0.5
        
    return np.clip(tau1, 0.0, 1.0), np.clip(tau2, 0.0, 1.0)

def apply_selective_thresholds(probs: np.ndarray, tau1: float, tau2: float) -> np.ndarray:
    decisions = np.full(len(probs), 2, dtype=int)
    decisions[probs < tau1] = 0
    decisions[probs >= tau2] = 1
    return decisions

def compute_confusion_matrix_3x2(targets: np.ndarray, probs: np.ndarray, 
                                 tau1: float, tau2: float) -> Dict[str, int]:

    decisions = apply_selective_thresholds(probs, tau1, tau2)
    
    n_neg = int(np.sum(targets == 0))
    n_pos = int(np.sum(targets == 1))
    
    TN = int(np.sum((decisions == 0) & (targets == 0)))
    RN_neg = int(np.sum((decisions == 2) & (targets == 0)))
    FP = int(np.sum((decisions == 1) & (targets == 0)))
    
    FN = int(np.sum((decisions == 0) & (targets == 1)))
    RP = int(np.sum((decisions == 2) & (targets == 1)))
    TP = int(np.sum((decisions == 1) & (targets == 1)))
    
    return {
        'TN': TN, 'RN_neg': RN_neg, 'FP': FP,
        'FN': FN, 'RP': RP, 'TP': TP,
        'n_neg': n_neg, 'n_pos': n_pos
    }

def compute_rates_from_confusion(cm: Dict[str, int], normalize: str = 'by_class') -> Dict[str, float]:
    if normalize == 'by_class':
        tnr = cm['TN'] / cm['n_neg'] if cm['n_neg'] > 0 else 0
        rnr = cm['RN_neg'] / cm['n_neg'] if cm['n_neg'] > 0 else 0
        fpr = cm['FP'] / cm['n_neg'] if cm['n_neg'] > 0 else 0
        fnr = cm['FN'] / cm['n_pos'] if cm['n_pos'] > 0 else 0
        rpr = cm['RP'] / cm['n_pos'] if cm['n_pos'] > 0 else 0
        tpr = cm['TP'] / cm['n_pos'] if cm['n_pos'] > 0 else 0
    else:
        n_total = cm['n_neg'] + cm['n_pos']
        tnr = cm['TN'] / n_total
        rnr = cm['RN_neg'] / n_total
        fpr = cm['FP'] / n_total
        fnr = cm['FN'] / n_total
        rpr = cm['RP'] / n_total
        tpr = cm['TP'] / n_total
        
    return {
        'TNR': tnr, 'RNR': rnr, 'FPR': fpr,
        'FNR': fnr, 'RPR': rpr, 'TPR': tpr,
    }

def calculate_sample_uncertainty(probs: np.ndarray) -> np.ndarray:
    return 1 - 2 * np.abs(probs - 0.5)

def calculate_true_cost_standard(targets: np.ndarray, decisions: np.ndarray, 
                                 cost_fn: float, cost_fp: float, 
                                 cRP: float, cRN: float) -> float:
    total_cost = 0.0
    for i in range(len(targets)):
        y = int(targets[i])
        d = decisions[i]
        if d == 2: # Reject
            total_cost += cRP if y == 1 else cRN
        elif d == y:
            total_cost += 0
        elif y == 1:
            total_cost += cost_fn
        else:
            total_cost += cost_fp
    return total_cost

def calculate_true_cost_removed(targets: np.ndarray, decisions: np.ndarray, 
                                cost_fn: float, cost_fp: float) -> float:
    total_cost = 0.0
    for i in range(len(targets)):
        y = int(targets[i])
        d = decisions[i]
        if d == 2:
            total_cost += 0
        elif d == y:
            total_cost += 0
        elif y == 1:
            total_cost += cost_fn
        else:
            total_cost += cost_fp
    return total_cost

def generate_cost_coverage_curve(probs: np.ndarray, targets: np.ndarray, 
                                 cost_fn: float, cost_fp: float, C_ref: float,
                                 use_baseline_norm: bool = True) -> pd.DataFrame:

    
    n = len(probs)
    cRP, cRN = derive_rejection_costs(cost_fn, cost_fp, C_ref)
    uncertainties = calculate_sample_uncertainty(probs)
    sorted_indices = np.argsort(-uncertainties)
    
    # Initialize decisions to classify all
    decisions = np.full(n, 2, dtype=int) 
    # Actually, standard approach: Start classified, then reject most uncertain
    decisions = np.array([1 if p >= 0.5 else 0 for p in probs], dtype=int)
    
    results = []
    
    # 100% Coverage Point
    total_cost_std = calculate_true_cost_standard(targets, decisions, cost_fn, cost_fp, cRP, cRN)
    total_cost_rem = calculate_true_cost_removed(targets, decisions, cost_fn, cost_fp)
    
    baseline_std = total_cost_std / n if n > 0 else 1.0
    baseline_rem = total_cost_rem / n if n > 0 else 1.0
    
    results.append({
        'coverage': 1.0,
        'total_true_cost_standard': total_cost_std,
        'total_true_cost_removed': total_cost_rem,
        'avg_true_cost_standard': baseline_std,
        'avg_true_cost_removed': baseline_rem,
        'avg_true_cost_standard_norm': 1.0,
        'avg_true_cost_removed_norm': 1.0,
        'total_true_cost_removed_norm': 1.0,
    })
    
    # Iteratively reject
    current_decisions = decisions.copy()
    for step, idx in enumerate(sorted_indices, 1):
        current_decisions[idx] = 2 # Reject
        n_classified = n - step
        
        tc_std = calculate_true_cost_standard(targets, current_decisions, cost_fn, cost_fp, cRP, cRN)
        tc_rem = calculate_true_cost_removed(targets, current_decisions, cost_fn, cost_fp)
        
        avg_std = tc_std / n
        avg_rem = tc_rem / n_classified if n_classified > 0 else 0
        
        norm_avg_std = avg_std / baseline_std if baseline_std > 0 else 1.0
        norm_avg_rem = avg_rem / baseline_rem if baseline_rem > 0 else 1.0
        norm_tot_rem = tc_rem / (baseline_rem * n) if baseline_rem > 0 else 1.0
        
        results.append({
            'coverage': (n - step) / n,
            'total_true_cost_standard': tc_std,
            'total_true_cost_removed': tc_rem,
            'avg_true_cost_standard': avg_std,
            'avg_true_cost_removed': avg_rem,
            'avg_true_cost_standard_norm': norm_avg_std,
            'avg_true_cost_removed_norm': norm_avg_rem,
            'total_true_cost_removed_norm': norm_tot_rem,
        })
        
    return pd.DataFrame(results)