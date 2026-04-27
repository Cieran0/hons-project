import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from src.utils.cache import InferenceCache

def main(args):
    # Load configs
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.calibration) as f:
        calib = json.load(f)
    temperature = calib['temperature']
    threshold = calib['threshold']
    print(f"Temperature: {temperature:.4f}")
    print(f"Threshold: {threshold:.4f}")
    
    # Load cached TEST inference
    cache = InferenceCache(cache_dir=args.cache_dir)
    probs, targets, image_ids, _ = cache.load(
        args.checkpoint, args.manifest, 'test', temperature=temperature
    )
    if probs is None:
        print("No cached test inference found! Run run_inference.py --all_splits first.")
        return
    
    # Calculate metrics
    preds = (probs >= threshold).astype(int)
    roc_auc = roc_auc_score(targets, probs)
    precision, recall, _ = precision_recall_curve(targets, probs)
    pr_auc = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    metrics = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0,
        'accuracy': float((tp + tn) / len(targets)),
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total_samples': int(len(targets)),
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f" TEST SET EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['true_positives']}  FP: {metrics['false_positives']}")
    print(f"  FN: {metrics['false_negatives']}  TN: {metrics['true_negatives']}")
    print(f"{'='*60}")
    
    exp_id = manifest['split_id']
    results_dir = Path(args.output_dir) / exp_id / 'evaluation'
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'test_evaluation.json', 'w') as f:
        json.dump({'metrics': metrics, 'calibration_params': calib}, f, indent=2)
    print(f" Results saved to {results_dir / 'test_evaluation.json'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Model (Uses Cached Inference)')
    parser.add_argument('--catalog', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--calibration', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--output_dir', type=str, default='experiments')
    args = parser.parse_args()
    main(args)