"""
Parameter Importance Analysis for Distributed ZNE
Quantifies which parameters (partition count, noise, depth, algorithm family) 
have the strongest effect on error reduction.

Uses multiple statistical approaches:
1. Variance explained (eta-squared)
2. ANOVA F-statistics
3. Correlation coefficients
4. Effect sizes

Usage: python parameter_importance.py results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

def calculate_eta_squared(groups, overall_mean):
    """
    Calculate eta-squared (η²): proportion of variance explained by a factor.
    η² = SS_between / SS_total
    
    Returns value between 0 (no effect) and 1 (explains all variance)
    """
    # Between-group variance
    ss_between = sum(len(g) * (np.mean(g) - overall_mean)**2 for g in groups)
    
    # Total variance
    all_values = np.concatenate(groups)
    ss_total = sum((x - overall_mean)**2 for x in all_values)
    
    if ss_total == 0:
        return 0
    
    return ss_between / ss_total


def analyze_categorical_parameter(df, parameter, target='error_reduction'):
    """
    Analyze importance of a categorical parameter using ANOVA and effect size.
    
    Returns:
        eta_squared: Proportion of variance explained (0-1)
        f_statistic: ANOVA F-statistic
        p_value: Statistical significance
        n_groups: Number of categories
    """
    # Get groups
    groups = [df[df[parameter] == level][target].dropna().values 
              for level in df[parameter].unique()]
    
    # Remove empty groups
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) < 2:
        return {'eta_squared': 0, 'f_statistic': 0, 'p_value': 1, 'n_groups': len(groups)}
    
    # ANOVA
    f_stat, p_val = f_oneway(*groups)
    
    # Eta-squared
    overall_mean = df[target].mean()
    eta_sq = calculate_eta_squared(groups, overall_mean)
    
    return {
        'eta_squared': eta_sq,
        'f_statistic': f_stat,
        'p_value': p_val,
        'n_groups': len(groups)
    }


def analyze_continuous_parameter(df, parameter, target='error_reduction'):
    """
    Analyze importance of a continuous parameter using correlation.
    
    Returns:
        r_squared: Proportion of variance explained by linear relationship
        correlation: Pearson correlation coefficient
        p_value: Statistical significance
    """
    # Remove NaN values
    data = df[[parameter, target]].dropna()
    
    if len(data) < 3:
        return {'r_squared': 0, 'correlation': 0, 'p_value': 1}
    
    # Pearson correlation
    corr, p_val = stats.pearsonr(data[parameter], data[target])
    
    return {
        'r_squared': corr**2,
        'correlation': corr,
        'p_value': p_val
    }


def comprehensive_parameter_analysis(df, output_dir='figures'):
    """
    Perform comprehensive parameter importance analysis.
    """
    print("\n" + "="*80)
    print("  PARAMETER IMPORTANCE ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Categorical parameters
    categorical_params = {
        'num_partitions_tested': 'Partition Count',
        'algorithm_family': 'Algorithm Family',
        'origin': 'Specific Algorithm'
    }
    
    # Continuous parameters
    continuous_params = {
        'local_noise': 'Local Noise Level',
        'circuit_depth': 'Circuit Depth',
        'partitioned_depth': 'Partitioned Depth'
    }
    
    print("\n--- Categorical Parameters (Variance Explained via η²) ---")
    for param, label in categorical_params.items():
        if param in df.columns:
            result = analyze_categorical_parameter(df, param)
            results[label] = {
                'type': 'categorical',
                'variance_explained': result['eta_squared'],
                'f_statistic': result['f_statistic'],
                'p_value': result['p_value'],
                'n_categories': result['n_groups']
            }
            
            print(f"\n{label}:")
            print(f"  Variance Explained (η²): {result['eta_squared']*100:.2f}%")
            print(f"  F-statistic: {result['f_statistic']:.2f}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Significant: {'Yes' if result['p_value'] < 0.05 else 'No'}")
            print(f"  Categories: {result['n_groups']}")
    
    print("\n--- Continuous Parameters (Variance Explained via R²) ---")
    for param, label in continuous_params.items():
        if param in df.columns:
            result = analyze_continuous_parameter(df, param)
            results[label] = {
                'type': 'continuous',
                'variance_explained': result['r_squared'],
                'correlation': result['correlation'],
                'p_value': result['p_value']
            }
            
            print(f"\n{label}:")
            print(f"  Variance Explained (R²): {result['r_squared']*100:.2f}%")
            print(f"  Correlation: {result['correlation']:.4f}")
            print(f"  p-value: {result['p_value']:.4f}")
            print(f"  Significant: {'Yes' if result['p_value'] < 0.05 else 'No'}")
    
    # Create ranking
    print("\n" + "="*80)
    print("  PARAMETER IMPORTANCE RANKING")
    print("="*80)
    
    ranking = sorted(results.items(), 
                    key=lambda x: x[1]['variance_explained'], 
                    reverse=True)
    
    print("\nRanked by variance explained:")
    for i, (param, data) in enumerate(ranking, 1):
        print(f"{i}. {param}: {data['variance_explained']*100:.2f}%")
    
    # Visualization
    Path(output_dir).mkdir(exist_ok=True)
    
    # Bar chart of variance explained
    fig, ax = plt.subplots(figsize=(10, 6))
    
    params = [r[0] for r in ranking]
    variance = [r[1]['variance_explained'] * 100 for r in ranking]
    colors = ['#ff6b6b' if r[1]['p_value'] < 0.05 else '#95e1d3' for r in ranking]
    
    bars = ax.barh(params, variance, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax.set_title('Parameter Importance for Error Reduction\n(Higher = More Important)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, variance)):
        ax.text(val + 0.5, i, f'{val:.1f}%', 
               va='center', fontsize=10, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff6b6b', alpha=0.8, label='Significant (p < 0.05)'),
        Patch(facecolor='#95e1d3', alpha=0.8, label='Not Significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    filepath = Path(output_dir) / 'parameter_importance_ranking.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filepath}")
    plt.close()
    
    return results, ranking


def strategy_specific_analysis(df, output_dir='figures'):
    """
    Analyze parameter importance separately for each strategy.
    """
    print("\n" + "="*80)
    print("  STRATEGY-SPECIFIC PARAMETER ANALYSIS")
    print("="*80)
    
    if 'strategy' not in df.columns:
        print("No 'strategy' column found, skipping")
        return
    
    strategies = sorted(df['strategy'].unique())
    
    fig, axes = plt.subplots(1, len(strategies), figsize=(6*len(strategies), 6), sharey=True)
    if len(strategies) == 1:
        axes = [axes]
    
    for ax, strategy in zip(axes, strategies):
        df_strat = df[df['strategy'] == strategy]
        
        print(f"\n--- {strategy.upper()} Strategy ---")
        
        # Analyze key parameters
        params_to_test = {
            'num_partitions_tested': 'Partitions',
            'local_noise': 'Noise',
            'algorithm_family': 'Algorithm'
        }
        
        results = []
        labels = []
        
        for param, label in params_to_test.items():
            if param not in df_strat.columns:
                continue
                
            if param in ['num_partitions_tested', 'algorithm_family']:
                result = analyze_categorical_parameter(df_strat, param)
                variance = result['eta_squared'] * 100
            else:
                result = analyze_continuous_parameter(df_strat, param)
                variance = result['r_squared'] * 100
            
            results.append(variance)
            labels.append(label)
            print(f"  {label}: {variance:.2f}%")
        
        # Plot
        bars = ax.barh(labels, results, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Variance Explained (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{strategy.upper()} Strategy', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, results)):
            ax.text(val + 0.5, i, f'{val:.1f}%', 
                   va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    filepath = Path(output_dir) / 'parameter_importance_by_strategy.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filepath}")
    plt.close()


def interaction_effects_analysis(df):
    """
    Analyze interaction effects between partition count and noise.
    """
    print("\n" + "="*80)
    print("  INTERACTION EFFECTS")
    print("="*80)
    
    if 'num_partitions_tested' not in df.columns or 'local_noise' not in df.columns:
        print("Missing required columns, skipping")
        return
    
    # Create interaction term
    df['partition_noise_interaction'] = df['num_partitions_tested'] * df['local_noise']
    
    # Analyze interaction
    result = analyze_continuous_parameter(df, 'partition_noise_interaction')
    
    print(f"\nPartition × Noise Interaction:")
    print(f"  Variance Explained (R²): {result['r_squared']*100:.2f}%")
    print(f"  Correlation: {result['correlation']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")
    
    if result['r_squared'] > 0.05:
        print("\n  ⚠ Strong interaction detected!")
        print("    The effect of partitions depends on noise level (or vice versa)")
    else:
        print("\n  ✓ Weak interaction - parameters act independently")


def main():
    """Main analysis pipeline"""
    
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../results.csv'
    
    print("\n" + "="*80)
    print("  LOADING DATA")
    print("="*80)
    
    try:
        df_full = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df_full)} experiments from {input_file}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nUsage: python parameter_importance.py [path_to_results.csv]")
        sys.exit(1)
    
    # Filter for TP primitive only (as specified by user)
    if 'communication_primitive' in df_full.columns:
        df = df_full[df_full['communication_primitive'] == 'tp'].copy()
        print(f"✓ Filtered to TP primitive only: {len(df)} experiments")
    else:
        df = df_full.copy()
        print("⚠ No communication_primitive column found, using all data")
    
    # Filter out partition=1 (no distribution)
    if 'num_partitions_tested' in df.columns:
        df = df[df['num_partitions_tested'] > 1].copy()
        print(f"✓ Filtered to partitions > 1: {len(df)} experiments")
    
    # Remove NaN in error_reduction
    df = df.dropna(subset=['error_reduction'])
    print(f"✓ After removing NaN: {len(df)} experiments")
    
    # Add algorithm family if not present
    if 'algorithm_family' not in df.columns and 'origin' in df.columns:
        df['algorithm_family'] = df['origin'].str.extract(r'([a-z]+)_')[0]
        print(f"✓ Extracted algorithm families: {df['algorithm_family'].nunique()} unique")
    
    print(f"\nFinal dataset: {len(df)} experiments for analysis")
    
    # Run analyses
    results, ranking = comprehensive_parameter_analysis(df)
    strategy_specific_analysis(df)
    interaction_effects_analysis(df)
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    top_3 = ranking[:3]
    print("\nTop 3 most important parameters:")
    for i, (param, data) in enumerate(top_3, 1):
        print(f"  {i}. {param}: {data['variance_explained']*100:.1f}% variance explained")
    
    total_variance = sum(r[1]['variance_explained'] for r in ranking)
    print(f"\nTotal variance explained by all parameters: {total_variance*100:.1f}%")
    print(f"Unexplained variance: {(1-total_variance)*100:.1f}%")
    
    print("\n✓ Analysis complete!")
    print("\nGenerated files in 'figures/' directory:")
    for fig in sorted(Path('figures').glob('parameter_*.png')):
        print(f"  - {fig.name}")


if __name__ == "__main__":
    main()