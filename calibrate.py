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
from tqdm import tqdm
from sklearn.metrics import roc_curve, confusion_matrix
from src.utils.cache import InferenceCache

def calibrate_temperature(probs, targets, device):
    """Optimise temperature parameter."""
    print("Optimising temperature...")
    class TemperatureScaling(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = torch.nn.Parameter(torch.ones(1) * 1.0)
        def forward(self, logits):
            return logits / self.temperature
    
    # Convert probs back to logits
    logits = torch.logit(torch.from_numpy(probs).clip(1e-7, 1-1e-7))
    targets = torch.from_numpy(targets)
    scaler = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    logits = logits.to(device)
    targets = targets.to(device)
    
    def eval_loss():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scaler(logits), targets
        )
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    temp = scaler.temperature.item()
    print(f" Optimal temperature: {temp:.4f}")
    return temp

def find_optimal_threshold(probs, targets, temperature, target_specificity=0.95):
    """Find threshold for target specificity."""
    print(f" Finding threshold for {target_specificity*100:.0f}% specificity...")
    # Apply temperature
    logits = torch.logit(torch.from_numpy(probs).clip(1e-7, 1-1e-7))
    calibrated_probs = torch.sigmoid(logits / temperature).numpy()
    fpr, tpr, thresholds = roc_curve(targets, calibrated_probs)
    specificities = 1 - fpr
    valid_idx = np.where(specificities >= target_specificity)[0]
    if len(valid_idx) == 0:
        threshold = 0.5
    else:
        best_idx = valid_idx[np.argmax(tpr[valid_idx])]
        threshold = thresholds[best_idx]
    preds = (calibrated_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f" Optimal threshold: {threshold:.4f}")
    print(f"   Sensitivity: {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    return threshold, sensitivity, specificity

def main(args):
    # Load configs
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.config) as f:
        train_config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load cached inference
    cache = InferenceCache(cache_dir=args.cache_dir)
    probs, targets, image_ids, meta = cache.load(
        args.checkpoint, args.manifest, 'val', temperature=None
    )
    if probs is None:
        print("No cached inference found! Run run_inference.py first.")
        print("   python src/run_inference.py --catalog ... --manifest ... --checkpoint ...")
        return
    
    # Calibrate
    temperature = calibrate_temperature(probs, targets, device)
    threshold, sensitivity, specificity = find_optimal_threshold(
        probs, targets, temperature, args.target_specificity
    )
    
    exp_id = manifest['split_id']  # e.g., 'exp_001'
    results_dir = Path(args.output_dir) / exp_id / 'calibration'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    calib_params = {
        'temperature': float(temperature),
        'threshold': float(threshold),
        'sensitivity_at_threshold': float(sensitivity),
        'specificity_at_threshold': float(specificity),
        'target_specificity': args.target_specificity,
    }
    with open(results_dir / 'calibration.json', 'w') as f:
        json.dump(calib_params, f, indent=2)
    print(f" Calibration saved to {results_dir / 'calibration.json'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calibrate Model (Uses Cached Inference)')
    parser.add_argument('--catalog', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='cache')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--target_specificity', type=float, default=0.95)
    args = parser.parse_args()
    main(args)