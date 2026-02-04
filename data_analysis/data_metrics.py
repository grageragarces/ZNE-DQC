"""
Investigate whether "higher noise → better error reduction" is an artifact
of using relative vs absolute error metrics.

This script checks:
1. Does higher network noise correlate with better error_reduction (relative)?
2. Does higher network noise correlate with worse zne_error (absolute)?
3. Is the "counterintuitive" pattern just a metric interpretation issue?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_metric_artifact(df):
    """
    Check if the counterintuitive noise pattern is due to metric choice.
    """
    
    # Filter to distributed data only (partition > 1)
    if 'num_partitions_tested' in df.columns:
        df = df[df['num_partitions_tested'] > 1].copy()
        print(f"\nUsing {len(df)} experiments (partition > 1 only)")
    
    # CRITICAL: Remove extreme outliers in error_reduction (same as analyze_results.py)
    if 'error_reduction' in df.columns:
        before = len(df)
        print(f"\n--- Data Quality Check ---")
        print(f"  Original error_reduction range: [{df['error_reduction'].min():.4f}, {df['error_reduction'].max():.4f}]")
        extreme_low = (df['error_reduction'] < -2.0).sum()
        extreme_high = (df['error_reduction'] > 2.0).sum()
        print(f"    Extreme negative (<-2.0): {extreme_low} ({extreme_low/len(df)*100:.1f}%)")
        print(f"    Extreme positive (>2.0): {extreme_high} ({extreme_high/len(df)*100:.1f}%)")
        
        df = df[(df['error_reduction'] >= -2.0) & (df['error_reduction'] <= 2.0)].copy()
        after = len(df)
        print(f"  Filtered out {before - after} extreme outliers")
        print(f"  Retained: {after}/{before} ({after/before*100:.1f}%)")
    
    if 'communication_noise_multiplier' not in df.columns:
        print("ERROR: 'communication_noise_multiplier' column not found")
        return
    
    # Group by network noise level
    noise_groups = df.groupby('communication_noise_multiplier').agg({
        'error_reduction': ['mean', 'std', 'count'],  # RELATIVE metric
        'zne_error': ['mean', 'std'],                 # ABSOLUTE metric  
        'noisy_error': ['mean', 'std']                # Baseline
    }).round(4)
    
    print("\n--- Performance by Network Noise Level ---")
    print(noise_groups)
    
    # Correlation analysis
    print("\n--- Correlation Analysis ---")
    
    # 1. Network noise vs error_reduction (relative improvement)
    corr_relative, p_relative = stats.spearmanr(
        df['communication_noise_multiplier'], 
        df['error_reduction']
    )
    print(f"\n1. Network Noise vs Error Reduction (RELATIVE):")
    print(f"   Spearman ρ = {corr_relative:.4f}, p = {p_relative:.4e}")
    if corr_relative > 0:
        print(f"   → Higher noise correlates with BETTER error_reduction ⚠️")
    else:
        print(f"   → Higher noise correlates with worse error_reduction")
    
    # 2. Network noise vs zne_error (absolute final error)
    corr_absolute, p_absolute = stats.spearmanr(
        df['communication_noise_multiplier'],
        df['zne_error']
    )
    print(f"\n2. Network Noise vs ZNE Error (ABSOLUTE):")
    print(f"   Spearman ρ = {corr_absolute:.4f}, p = {p_absolute:.4e}")
    if corr_absolute > 0:
        print(f"   → Higher noise correlates with WORSE final error ✓")
    else:
        print(f"   → Higher noise correlates with better final error (unexpected!)")
    
    # 3. Network noise vs baseline noisy_error
    corr_baseline, p_baseline = stats.spearmanr(
        df['communication_noise_multiplier'],
        df['noisy_error']
    )
    print(f"\n3. Network Noise vs Noisy Error (BASELINE):")
    print(f"   Spearman ρ = {corr_baseline:.4f}, p = {p_baseline:.4e}")
    if corr_baseline > 0:
        print(f"   → Higher noise correlates with WORSE baseline ✓")
    
    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    if corr_relative > 0 and corr_absolute > 0 and corr_baseline > 0:
        print("\n✓ METRIC ARTIFACT CONFIRMED!")
        print("\nThe pattern is:")
        print("  - Higher network noise → worse baseline (noisy_error ↑)")
        print("  - Higher network noise → worse final result (zne_error ↑)")
        print("  - BUT: Higher network noise → better relative improvement (error_reduction ↑)")
        print("\nThis is because:")
        print("  error_reduction = (noisy_error - zne_error) / noisy_error")
        print("  When baseline is worse, there's 'more room' for relative improvement!")
        print("\n⚠️  WARNING: 'Better error_reduction' does NOT mean 'better performance'")
        print("   It means ZNE works harder on a worse baseline, but still delivers worse results.")
        
    elif corr_relative > 0 and corr_absolute < 0:
        print("\n❓ UNEXPECTED: Higher noise yields both better relative AND absolute results")
        print("   This would be genuinely counterintuitive and worth investigating further.")
        
    else:
        print("\n✓ NO ARTIFACT: Results align with expectations")
    
    return {
        'corr_relative': corr_relative,
        'corr_absolute': corr_absolute,
        'corr_baseline': corr_baseline,
        'noise_groups': noise_groups
    }

def plot_relative_vs_absolute(df, output_dir='figures'):
    """
    Create visualization showing the relative vs absolute metric issue.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if 'communication_noise_multiplier' not in df.columns:
        print("Skipping plot: missing required columns")
        return
    
    # Filter to distributed data
    if 'num_partitions_tested' in df.columns:
        df = df[df['num_partitions_tested'] > 1].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Error Reduction (Relative) vs Network Noise
    ax = axes[0]
    grouped = df.groupby('communication_noise_multiplier').agg({
        'error_reduction': ['mean', 'std', 'count']
    })
    x = grouped.index
    y = grouped['error_reduction']['mean']
    err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
    
    ax.plot(x, y, 'o-', linewidth=2.5, markersize=10, color='#2ecc71', label='Error Reduction')
    ax.fill_between(x, y-err, y+err, alpha=0.3, color='#2ecc71')
    ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Reduction (Relative)', fontsize=12, fontweight='bold')
    ax.set_title('RELATIVE Improvement', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: ZNE Error (Absolute) vs Network Noise  
    ax = axes[1]
    grouped = df.groupby('communication_noise_multiplier').agg({
        'zne_error': ['mean', 'std', 'count']
    })
    x = grouped.index
    y = grouped['zne_error']['mean']
    err = grouped['zne_error']['std'] / np.sqrt(grouped['zne_error']['count'])
    
    ax.plot(x, y, 's-', linewidth=2.5, markersize=10, color='#e74c3c', label='ZNE Error')
    ax.fill_between(x, y-err, y+err, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
    ax.set_ylabel('ZNE Error (Absolute)', fontsize=12, fontweight='bold')
    ax.set_title('ABSOLUTE Final Error', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Both baselines for comparison
    ax = axes[2]
    grouped_noisy = df.groupby('communication_noise_multiplier')['noisy_error'].mean()
    grouped_zne = df.groupby('communication_noise_multiplier')['zne_error'].mean()
    
    x = grouped_noisy.index
    ax.plot(x, grouped_noisy.values, '^-', linewidth=2.5, markersize=10, 
            color='#95a5a6', label='Baseline (noisy)', alpha=0.7)
    ax.plot(x, grouped_zne.values, 's-', linewidth=2.5, markersize=10,
            color='#3498db', label='After ZNE')
    ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (Absolute)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline vs Mitigated Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Relative vs Absolute Metrics: Understanding the "Counterintuitive" Pattern',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, 'metric_artifact_investigation.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {filepath}")
    plt.close()
    
    # Additional plot: Show the metric artifact explicitly
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sample a few noise levels for clarity
    noise_levels = sorted(df['communication_noise_multiplier'].unique())
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(noise_levels)))
    
    for i, noise in enumerate(noise_levels):
        df_noise = df[df['communication_noise_multiplier'] == noise]
        
        # Calculate mean values
        mean_noisy = df_noise['noisy_error'].mean()
        mean_zne = df_noise['zne_error'].mean()
        mean_reduction = df_noise['error_reduction'].mean()
        
        # Plot bars showing baseline and ZNE error
        ax.bar(i - 0.15, mean_noisy, width=0.3, color=colors[i], alpha=0.4, 
               label=f'Noise={noise}' if i == 0 else '')
        ax.bar(i + 0.15, mean_zne, width=0.3, color=colors[i], alpha=0.9)
        
        # Annotate with error_reduction percentage
        ax.text(i, max(mean_noisy, mean_zne) * 1.05, 
                f'{mean_reduction:.1%}', 
                ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f'{n:.2f}' for n in noise_levels])
    ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error', fontsize=12, fontweight='bold')
    ax.set_title('Why "Better Error Reduction" ≠ "Better Performance"\n(Percentages show relative improvement)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.4, label='Baseline (noisy_error)'),
        Patch(facecolor='gray', alpha=0.9, label='After ZNE (zne_error)'),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc='upper left', frameon=True, shadow=True)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'metric_artifact_explained.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved explanation: {filepath}")
    plt.close()

def main():
    import sys
    
    # Load data
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../results.csv'
    
    print(f"Loading data from: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} experiments")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nUsage: python investigate_metrics.py [path_to_results.csv]")
        sys.exit(1)
    
    # Run analysis
    results = analyze_metric_artifact(df)
    
    # Create visualizations
    plot_relative_vs_absolute(df)

if __name__ == "__main__":
    main()