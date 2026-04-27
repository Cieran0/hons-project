import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


DERM_TRIAGE_COSTS = {
    'cost_fp': 750.0,
    'cost_referral': 150.0,
    
    'fn_tiers': {
        '2_low': {
            'cost_fn': 2000.0,
            'description': '',
            'clinical_scenario': '',
            'evidence': ''
        },
        '3_moderate': {
            'cost_fn': 7500.0,
            'description': '',
            'clinical_scenario': '',
            'evidence': ''
        },
        '4_high': {
            'cost_fn': 50000.0,
            'description': '',
            'clinical_scenario': '',
            'evidence': ''
        },
        '5_most_conservative': {
            'cost_fn': 100000.0,
            'description': '',
            'clinical_scenario': '',
            'evidence': ''
        }
    }
}


def generate_real_life_configs(
    fn_tier_names: List[str] = None,
    include_metadata: bool = True,
    normalize_by_fn: bool = True
) -> List[Dict[str, Any]]:

    if fn_tier_names is None:
        fn_tier_names = list(DERM_TRIAGE_COSTS['fn_tiers'].keys())
    
    configs = []
    config_id = 0
    
    cost_fp = DERM_TRIAGE_COSTS['cost_fp']
    cost_referral = DERM_TRIAGE_COSTS['cost_referral']
    
    for tier_name in fn_tier_names:
        tier = DERM_TRIAGE_COSTS['fn_tiers'][tier_name]
        cost_fn = tier['cost_fn']
        
        # Clinical validity check
        if cost_fn < cost_fp:
            print(f"️  Warning: Tier {tier_name} has cost_fn ({cost_fn}) < cost_fp ({cost_fp})")
        
        config_id += 1
        
        cost_r = cost_referral
        
        # Normalize costs by cFN for comparability across tiers
        if normalize_by_fn:
            cost_fn_norm = 1.0
            cost_fp_norm = cost_fp / cost_fn
            cost_r_norm = cost_r / cost_fn
        else:
            cost_fn_norm = cost_fn
            cost_fp_norm = cost_fp
            cost_r_norm = cost_r
        
        fp_fn_name = f"1:{int(cost_fn / cost_fp)}"
        
        config = {
            'id': config_id,
            'cost_config': f"REAL:{tier_name}",
            'tier_name': tier_name,
            'fp_fn_name': fp_fn_name,
            'cR_ratio': cost_r / cost_fp,
            
            # Original costs (for threshold calculation)
            'cost_fn': cost_fn,
            'cost_fp': cost_fp,
            'cost_r': cost_r,
            'cost_referral_fixed': cost_referral,
            
            # Normalized costs (for comparison/plotting)
            'cost_fn_norm': cost_fn_norm,
            'cost_fp_norm': cost_fp_norm,
            'cost_r_norm': cost_r_norm,
            
            # Metadata
            'fp_fn_ratio': cost_fp / cost_fn,
            'tier_description': tier['description'],
            'clinical_scenario': tier['clinical_scenario'],
            'evidence': tier['evidence'],
        }
        
        if include_metadata:
            config['metadata'] = {
                'cheaper_error': cost_fp,
                'cR_percentage': f"{(cost_r / cost_fp)*100:.1f}%",
                'description': f"FN=£{cost_fn:,.0f}, FP=£{cost_fp:,.0f}, cR=£{cost_r:,.0f} (FIXED)",
                'normalized': f"FN=1.0, FP={cost_fp_norm:.3f}, cR={cost_r_norm:.3f}",
            }
        
        configs.append(config)
    
    return configs

def get_config_by_tier(
    tier_name: str,
    normalize_by_fn: bool = True
) -> Dict[str, Any]:
    """Get a single configuration for a specific FN tier."""
    configs = generate_real_life_configs(
        fn_tier_names=[tier_name],
        normalize_by_fn=normalize_by_fn
    )
    return configs[0] if configs else None


def save_real_life_configs(
    output_path: str,
    fn_tier_names: List[str] = None,
    include_summary: bool = True
):
    configs = generate_real_life_configs(
        fn_tier_names=fn_tier_names,
        include_metadata=True,
        normalize_by_fn=True
    )
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'configurations': configs,
        'num_configs': len(configs),
        'cost_structure': {
            'fixed': {
                'cost_fp': DERM_TRIAGE_COSTS['cost_fp'],
                'cost_referral': DERM_TRIAGE_COSTS['cost_referral'],
                'currency': 'GBP',
                'year': '2024-25',
                'source': '/Scotland health economics data',
                'note': 'C_ref is FIXED at £227 for all tiers (no cR_ratio variation)'
            },
            'fn_tiers': {k: {kk: vv for kk, vv in v.items() if kk != 'evidence'} 
                        for k, v in DERM_TRIAGE_COSTS['fn_tiers'].items()}
        }
    }
    
    if include_summary:
        fn_norm_values = [c['cost_fn_norm'] for c in configs]
        fp_norm_values = [c['cost_fp_norm'] for c in configs]
        cr_norm_values = [c['cost_r_norm'] for c in configs]
        
        output_data['summary'] = {
            'total_configurations': len(configs),
            'fn_tiers_included': list(set(c['tier_name'] for c in configs)),
            'c_ref_fixed': DERM_TRIAGE_COSTS['cost_referral'],
            
            'cost_fn_norm_range': [min(fn_norm_values), max(fn_norm_values)],
            'cost_fp_norm_range': [min(fp_norm_values), max(fp_norm_values)],
            'cost_r_norm_range': [min(cr_norm_values), max(cr_norm_values)],
            
            'interpretation': {
                'cost_fn_norm': 'Always 1.0 (baseline: cost of missing melanoma)',
                'cost_fp_norm': 'FP cost as fraction of FN cost',
                'cost_r_norm': 'Rejection cost as fraction of FN cost (C_ref=£227 fixed)',
                'recommended_tier': '3_moderate (£7,500 FN) for service planning',
                'c_ref_note': 'Fixed at £227 =  dermatologist appointment cost'
            }
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f" Saved {len(configs)} real-life configurations to {output_path}")

def load_real_life_configs(input_path: str) -> List[Dict]:
    """Load configs from JSON (compatible with existing load_configs)."""
    with open(input_path) as f:
        data = json.load(f)
    if 'configurations' in data:
        return data['configurations']
    else:
        return data

def main():
    parser = argparse.ArgumentParser(
        description='Generate Real-Worldish Dermatology Triage Cost Configs (FIXED C_ref=£227)'
    )
    parser.add_argument(
        '--output', type=str, 
        default='configs/real_life_dermatology_configs.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--tiers', type=str, nargs='+', default=None,
        help='Specific tier names to include (default: all 4)'
    )
    parser.add_argument(
        '--summary-only', action='store_true',
        help='Print clinical summary without saving'
    )
    parser.add_argument(
        '--no-normalize', action='store_true',
        help='Do not normalize costs by cFN (output raw £ values)'
    )
    args = parser.parse_args()
    
    
    normalize = not args.no_normalize
    
    if args.tiers:
        print(f" Generating custom real-life configuration subset...")
        configs = generate_real_life_configs(
            fn_tier_names=args.tiers,
            normalize_by_fn=normalize
        )
    else:
        print(f" Generating full real-life configuration set (4 FN tiers, FIXED C_ref=£227)...")
        configs = generate_real_life_configs(normalize_by_fn=normalize)
    
    save_real_life_configs(
        args.output,
        fn_tier_names=args.tiers,
        include_summary=True
    )
    
    print(f"\n Real-life configuration generation complete!")
    print(f"   Total configurations: {len(configs)}")
    print(f"   False Negative range: £1,650 to £225,000")
    print(f"   False Positive: £400 (fixed)")
    print(f"   Referral cost (C_ref): £227 (FIXED for all tiers) ← KEY CHANGE")
    print(f"   Minimal tier removed: clinically unrealistic for melanoma triage")
    if normalize:
        print(f"    All costs normalized by cFN for direct comparison")

if __name__ == '__main__':
    main()