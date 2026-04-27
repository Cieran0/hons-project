import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('seaborn-v0_8-whitegrid')

def plot_3x2_confusion(cm: Dict[str, int], config_name: str, coverage: float, 
                       output_path: Path, mode: str = 'counts', 
                       title_suffix: str = '') -> None:
    """Plot 3x2 confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    raw_matrix = np.array([
        [cm['TN'], cm['RN_neg'], cm['FP']],
        [cm['FN'], cm['RP'], cm['TP']],
    ])
    
    if mode == 'normalized':
        matrix = raw_matrix.astype(float)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums
        fmt, cmap = '.1%', 'Greens'
        cbar_label = 'Rate'
    else:
        matrix = raw_matrix.astype(int)
        fmt, cmap = 'd', 'Blues'
        cbar_label = 'Count'
        
    sns.heatmap(matrix, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=['Benign', 'Reject', 'Melanoma'],
                yticklabels=['Benign', 'Melanoma'],
                ax=ax, cbar_kws={'label': cbar_label})
    
    ax.set_xlabel('Decision', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    
    title = f'{config_name}\nCoverage: {coverage*100:.1f}% {title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cost_coverage_curves(results_dict: Dict[str, pd.DataFrame], 
                              output_path: Path, metric_col: str, 
                              title: str, xlim: Optional[Tuple] = None, 
                              ylim: Optional[Tuple] = None, 
                              normalize: bool = True) -> None:
    """Plot multiple cost-coverage curves."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(results_dict)))
    
    for idx, (name, df) in enumerate(results_dict.items()):
        col = f"{metric_col}_norm" if normalize and f"{metric_col}_norm" in df.columns else metric_col
        ax.plot(df['coverage'], df[col], label=name, linewidth=1.5, color=colors[idx % 20], alpha=0.7)
    
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (100%)')
    ax.set_xlabel('Coverage', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_col.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(ncol=2, fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cost_vs_cref_sweep(df_config: pd.DataFrame, output_path: Path, 
                            split_name: str, n_samples: int, config_name: str, 
                            c_ref_endpoint: float, is_positive: bool = True) -> None:
    """Plot Average True Cost vs C_ref."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df_config = df_config.sort_values('c_ref').copy()
    
    if is_positive:
        df_plot = df_config[df_config['c_ref'] <= c_ref_endpoint].copy()
    else:
        df_plot = df_config.copy()
    
    ax.plot(df_plot['c_ref'], df_plot['avg_true_cost'],
            linewidth=2.5, color='#E63946', alpha=0.95, label='Avg True Cost (Classified)')
    
    ax2 = ax.twinx()
    ax2.plot(df_plot['c_ref'], df_plot['coverage'],
             linewidth=1.5, color='#2E86AB', alpha=0.6, linestyle='--', label='Coverage')
    ax2.set_ylabel('Coverage', fontsize=10, color='#2E86AB')
    ax2.tick_params(axis='y', labelcolor='#2E86AB')
    ax2.set_ylim(0, 1)
    
    if is_positive:
        ax.set_xlim(0, c_ref_endpoint)
    else:
        ax.set_xlim(df_plot['c_ref'].min(), c_ref_endpoint)
        
    ax.set_xlabel('Referral Cost ($C_{ref}$)', fontsize=11)
    ax.set_ylabel('Avg True Cost (Classified Samples)', fontsize=11, color='#E63946')
    ax.tick_params(axis='y', labelcolor='#E63946')
    
    title = f'{config_name} | {split_name} (n={n_samples:,})'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_3d_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                    x_ratios: np.ndarray, y_ratios: np.ndarray, 
                    output_path: Path, metric_name: str, 
                    title_suffix: str = '', cmap: str = 'viridis',
                    elevation: float = 25, azimuth: float = 225,
                    z_lim: Optional[Tuple] = None) -> None:
    """Create 3D surface plot."""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')
    
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
    
    ax.set_xlabel('CFN / CFP Ratio', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('Cref / CFP Ratio', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_zlabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold', labelpad=15)
    
    title = f'{metric_name.replace("_", " ").title()} Surface{title_suffix}'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=25)
    
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=25, pad=0.1)
    cbar.set_label(metric_name.replace('_', ' ').title(), rotation=270, labelpad=25, fontsize=11)
    
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlim(x_ratios[0], x_ratios[-1])
    ax.set_ylim(y_ratios[0], y_ratios[-1])
    
    if z_lim:
        ax.set_zlim(z_lim[0], z_lim[1])
    else:
        z_margin = (Z.max() - Z.min()) * 0.1
        ax.set_zlim(Z.min() - z_margin, Z.max() + z_margin)
        
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_2d_contour(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                    x_ratios: np.ndarray, y_ratios: np.ndarray,
                    output_path: Path, metric_name: str,
                    title_suffix: str = '', cmap: str = 'viridis',
                    levels: int = 50) -> None:
    """Create 2D contour heatmap plot."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Contour fill
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, extend='both')
    
    # Contour lines
    cs = ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.3)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
    
    ax.set_xlabel('CFN / CFP Ratio (Cost of False Negative)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cref / CFP Ratio (Referral Cost)', fontsize=12, fontweight='bold')
    ax.set_title(f'2D Contour: {metric_name.replace("_", " ").title()}{title_suffix}', 
                 fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(metric_name.replace('_', ' ').title(), fontsize=11)
    
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()