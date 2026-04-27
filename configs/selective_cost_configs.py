# src/configs/selective_cost_configs.py (FILTERED + NORMALIZED)
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# ==========================================
# Generate FP:FN Ratios (FN-varies only)
# ==========================================
def generate_fp_fn_ratios(
    max_power: int = 20,
    base_cost: int = 1,
    fn_only: bool = True  # ✅ NEW: Only generate FN-varies ratios
) -> List[Dict[str, Any]]:
    """
    Generate FP:FN cost ratios.
    
    If fn_only=True: Only generate ratios where FN >= FP (1:1, 1:2, 1:4, ..., 1:2^20)
    If fn_only=False: Generate both directions (legacy behavior)
    """
    ratios = []
    
    # FN varies (FP=1): 1:1, 1:2, 1:4, ..., 1:2^31
    # These are the ONLY ratios we want for melanoma
    for power in range(0, max_power + 1):
        fn_cost = base_cost * (2 ** power)
        ratios.append({
            'name': f'1:{fn_cost}',
            'cost_fp': base_cost,
            'cost_fn': fn_cost,
            'ratio_type': 'fn_varies',
            'power': power,
            'fp_fn_ratio': base_cost / fn_cost,  # ✅ NEW: FP/FN ratio (<=1)
        })
    
    # Only add FP-varies if explicitly requested (legacy mode)
    if not fn_only:
        for power in range(max_power, 0, -1):
            fp_cost = base_cost * (2 ** power)
            ratios.append({
                'name': f'{fp_cost}:1',
                'cost_fp': fp_cost,
                'cost_fn': base_cost,
                'ratio_type': 'fp_varies',
                'power': power,
                'fp_fn_ratio': fp_cost / base_cost,  # >1 (filtered out later)
            })
    
    return ratios

# ==========================================
# Generate Cost Configs (Normalized)
# ==========================================
def generate_cost_configs(
    fp_fn_ratios: List[Dict] = None,
    cr_ratios: List[float] = None,
    include_metadata: bool = True,
    max_power: int = 31,
    fn_only: bool = True,  # ✅ NEW: Default to FN-only ratios
    normalize_by_fn: bool = True  # ✅ NEW: Normalize costs by cFN
) -> List[Dict[str, Any]]:
    """
    Generate cost configurations with filtering + normalization.
    
    - Filters out configs where cFP > cFN (clinically invalid for melanoma)
    - Normalizes all costs by cFN for direct comparability
    """
    # Use defaults if not provided
    if fp_fn_ratios is None:
        fp_fn_ratios = generate_fp_fn_ratios(max_power=max_power, fn_only=fn_only)
    if cr_ratios is None:
        cr_ratios = generate_cr_ratios()
    
    configs = []
    config_id = 0
    
    for ratio in fp_fn_ratios:
        # ✅ FILTER: Skip if FP cost > FN cost (clinically invalid)
        if ratio['cost_fp'] > ratio['cost_fn']:
            continue
            
        for cR_ratio in cr_ratios:
            config_id += 1
            
            # Base costs
            cost_fn = ratio['cost_fn']
            cost_fp = ratio['cost_fp']
            
            # Rejection cost (relative to cheaper error, which is now always FP)
            cheaper_error = cost_fp  # Since cFN >= cFP, FP is cheaper
            cost_r = cR_ratio * cheaper_error
            
            # ✅ NORMALIZE: Divide all costs by cFN for comparability
            if normalize_by_fn:
                cost_fn_norm = 1.0  # Always 1.0 by definition
                cost_fp_norm = cost_fp / cost_fn  # FP as fraction of FN
                cost_r_norm = cost_r / cost_fn  # Rejection as fraction of FN
            else:
                cost_fn_norm = cost_fn
                cost_fp_norm = cost_fp
                cost_r_norm = cost_r
            
            config = {
                'id': config_id,
                'cost_config': f"FP:FN={ratio['name']}",
                'cR_ratio': cR_ratio,
                
                # Original costs (for threshold calculation)
                'cost_fn': cost_fn,
                'cost_fp': cost_fp,
                'cost_r': cost_r,
                
                # ✅ Normalized costs (for comparison/plotting)
                'cost_fn_norm': cost_fn_norm,
                'cost_fp_norm': cost_fp_norm,
                'cost_r_norm': cost_r_norm,
                
                # Metadata
                'fp_fn_name': ratio['name'],
                'fp_fn_ratio_type': ratio.get('ratio_type', 'custom'),
                'fp_fn_ratio': ratio.get('fp_fn_ratio', cost_fp / cost_fn),  # FP/FN <= 1
            }
            
            if include_metadata:
                config['metadata'] = {
                    'cheaper_error': cheaper_error,
                    'cR_percentage': f"{cR_ratio*100:.0f}%",
                    'description': f"FN={cost_fn}, FP={cost_fp}, cR={cost_r:.2f}",
                    'normalized': f"FN=1.0, FP={cost_fp_norm:.3f}, cR={cost_r_norm:.3f}",
                }
            
            configs.append(config)
    
    return configs

# ==========================================
# Helper Functions
# ==========================================
def generate_cr_ratios(
    levels: List[float] = None,
    min_ratio: float = 0.0,
    max_ratio: float = 0.9,
    num_levels: int = 9
) -> List[float]:
    if levels is not None:
        return sorted(levels)
    return np.linspace(min_ratio, max_ratio, num_levels).tolist()

def generate_minimal_configs(
    fp_fn_ratios: List[str] = None,
    cr_ratios: List[float] = None,
    max_power: int = 31,
    fn_only: bool = True
) -> List[Dict[str, Any]]:
    """Generate minimal subset with filtering + normalization."""
    all_configs = generate_cost_configs(
        max_power=max_power,
        fn_only=fn_only,
        normalize_by_fn=True
    )
    
    # Filter by FP:FN ratios if specified
    if fp_fn_ratios:
        all_configs = [c for c in all_configs if c['fp_fn_name'] in fp_fn_ratios]
    
    # Filter by cR ratios if specified
    if cr_ratios:
        filtered = []
        for c in all_configs:
            for cr in cr_ratios:
                if abs(c['cR_ratio'] - cr) < 0.01:
                    filtered.append(c)
                    break
        all_configs = filtered
    
    return all_configs

def save_configs(configs: List[Dict], output_path: str, include_summary: bool = True):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'configurations': configs,
        'num_configs': len(configs),
    }
    
    if include_summary:
        fp_fn_unique = list(set(c['fp_fn_name'] for c in configs))
        cr_unique = sorted(list(set(c['cR_ratio'] for c in configs)))
        
        # Summary of normalized cost ranges
        fn_norm_values = [c['cost_fn_norm'] for c in configs]
        fp_norm_values = [c['cost_fp_norm'] for c in configs]
        cr_norm_values = [c['cost_r_norm'] for c in configs]
        
        output_data['summary'] = {
            'total_configurations': len(configs),
            'fp_fn_ratios': len(fp_fn_unique),
            'cr_ratios': len(cr_unique),
            'fp_fn_ratio_values': fp_fn_unique,
            'cr_ratio_values': cr_unique,
            
            # ✅ Normalized cost ranges (for interpretation)
            'cost_fn_norm_range': [min(fn_norm_values), max(fn_norm_values)],
            'cost_fp_norm_range': [min(fp_norm_values), max(fp_norm_values)],
            'cost_r_norm_range': [min(cr_norm_values), max(cr_norm_values)],
            
            # Clinical interpretation
            'interpretation': {
                'cost_fn_norm': 'Always 1.0 (baseline: cost of missing melanoma)',
                'cost_fp_norm': 'FP cost as fraction of FN cost (e.g., 0.01 = FP costs 1% of FN)',
                'cost_r_norm': 'Rejection cost as fraction of FN cost',
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Saved {len(configs)} configurations to {output_path}")
    if include_summary:
        print(f"\n📊 Configuration Summary (Normalized by cFN):")
        print(f"   Total configs: {output_data['summary']['total_configurations']}")
        print(f"   FP:FN ratios (FN>=FP): {output_data['summary']['fp_fn_ratios']}")
        print(f"   cR ratios: {output_data['summary']['cr_ratios']}")
        print(f"   FP cost range: {output_data['summary']['cost_fp_norm_range'][0]:.3f} to {output_data['summary']['cost_fp_norm_range'][1]:.3f} (relative to FN=1)")
        print(f"   cR cost range: {output_data['summary']['cost_r_norm_range'][0]:.3f} to {output_data['summary']['cost_r_norm_range'][1]:.3f} (relative to FN=1)")

def load_configs(input_path: str) -> List[Dict]:
    with open(input_path) as f:
        data = json.load(f)
    if 'configurations' in data:
        return data['configurations']
    else:
        return data

# ==========================================
# CLI Interface
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Generate Selective Classification Cost Configs (Filtered + Normalized)')
    parser.add_argument('--output', type=str, default='configs/selective_configs_normalized.json',
                        help='Output JSON file path')
    parser.add_argument('--max-power', type=int, default=20,
                        help='Maximum power for cost ratios (2^max_power)')
    parser.add_argument('--cr-levels', type=float, nargs='+', default=None,
                        help='Explicit cR ratio levels')
    parser.add_argument('--cr-min', type=float, default=0.1,
                        help='Minimum cR ratio')
    parser.add_argument('--cr-max', type=float, default=0.9,
                        help='Maximum cR ratio')
    parser.add_argument('--cr-num', type=int, default=9,
                        help='Number of cR levels')
    parser.add_argument('--minimal', action='store_true',
                        help='Generate minimal subset for testing')
    parser.add_argument('--no-metadata', action='store_true',
                        help='Exclude metadata from output')
    parser.add_argument('--allow-fp-gt-fn', action='store_true',  # ✅ NEW
                        help='Allow configs where FP cost > FN cost (legacy mode)')
    args = parser.parse_args()
    
    fn_only = not args.allow_fp_gt_fn
    
    if args.minimal:
        print("🔧 Generating minimal configuration subset (FN>=FP, normalized)...")
        configs = generate_minimal_configs(
            fp_fn_ratios=['1:1', '1:16', '1:64', '1:256'],
            cr_ratios=[0.3, 0.5, 0.7],
            max_power=args.max_power,
            fn_only=fn_only
        )
    else:
        print("🔧 Generating full configuration space (FN>=FP, normalized)...")
        fp_fn_ratios = generate_fp_fn_ratios(max_power=args.max_power, fn_only=fn_only)
        cr_ratios = generate_cr_ratios(
            levels=args.cr_levels,
            min_ratio=args.cr_min,
            max_ratio=args.cr_max,
            num_levels=args.cr_num
        )
        configs = generate_cost_configs(
            fp_fn_ratios=fp_fn_ratios,
            cr_ratios=cr_ratios,
            include_metadata=not args.no_metadata,
            max_power=args.max_power,
            fn_only=fn_only,
            normalize_by_fn=True  # ✅ Always normalize
        )
    
    save_configs(configs, args.output)
    print(f"\n✅ Configuration generation complete!")
    print(f"   Total configurations: {len(configs)}")
    print(f"   FP:FN range: 1:1 to 1:{2**args.max_power} (FN >= FP only)")
    print(f"   All costs normalized by cFN for direct comparison")

if __name__ == '__main__':
    main()