"""
ZNE Distributed Quantum Computing Analysis
Analyzes results from main_reduced.ipynb experiments

IMPORTANT: This script filters out partition=1 data from main analysis.
- Partition=1 means NO distribution (circuit runs as single unit)
- All strategies should be identical at partition=1
- Communication parameters have no effect at partition=1
- See fig0_baseline_validation_partition1.png to verify partition=1 behavior

Usage: python analyze_results.py ../results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 6)

def print_section(title):
    """Pretty print section headers"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def filter_distributed_data(df, keep_baseline=False):
    """
    Filter out partition=1 data (no distribution) from analysis.
    
    When partition=1, there is NO distribution - the circuit runs as a single unit.
    This means all strategies (global/local/no-ZNE) should produce identical results,
    and communication parameters (primitive, noise multiplier) have no effect.
    
    Args:
        df: DataFrame with experimental results
        keep_baseline: If True, return both filtered data and baseline separately
    
    Returns:
        df_distributed: Data with partition > 1 (actual distributed execution)
        df_baseline: Data with partition = 1 (optional, only if keep_baseline=True)
    """
    if 'num_partitions_tested' not in df.columns:
        print("⚠ Warning: 'num_partitions_tested' column not found, skipping filter")
        return (df, pd.DataFrame()) if keep_baseline else df
    
    partition_1_count = (df['num_partitions_tested'] == 1).sum()
    total_count = len(df)
    
    if partition_1_count > 0:
        print(f"\n--- Filtering Partition Data ---")
        print(f"  Partition = 1 (no distribution): {partition_1_count} experiments ({partition_1_count/total_count*100:.1f}%)")
        print(f"  Partition > 1 (distributed): {total_count - partition_1_count} experiments ({(total_count-partition_1_count)/total_count*100:.1f}%)")
        print(f"  → Excluding partition=1 from main analysis (no distribution occurs)")
        print(f"  → All strategies should be identical at partition=1")
    
    df_baseline = df[df['num_partitions_tested'] == 1].copy() if keep_baseline else pd.DataFrame()
    df_distributed = df[df['num_partitions_tested'] > 1].copy()
    
    if keep_baseline:
        return df_distributed, df_baseline
    return df_distributed

def load_and_clean_data(filepath):
    """Load and clean the experimental data"""
    print_section("LOADING DATA")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} experiments from {filepath}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nUsage: python analyze_results.py [path_to_results.csv]")
        sys.exit(1)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Display experimental parameters
    print("\n--- Experimental Parameters ---")
    if 'strategy' in df.columns:
        print(f"Strategies: {sorted(df['strategy'].unique())}")
    if 'communication_primitive' in df.columns:
        print(f"Communication primitives: {sorted(df['communication_primitive'].unique())}")
    if 'local_noise' in df.columns:
        print(f"Local noise levels: {sorted(df['local_noise'].unique())}")
    if 'communication_noise_multiplier' in df.columns:
        print(f"Comm noise multipliers: {sorted(df['communication_noise_multiplier'].unique())}")
    if 'num_partitions_tested' in df.columns:
        print(f"Partition range: {df['num_partitions_tested'].min()}-{df['num_partitions_tested'].max()}")
    if 'origin' in df.columns:
        print(f"Unique algorithms: {len(df['origin'].unique())}")
        print(f"Algorithms: {sorted(df['origin'].unique())}")
    
    # Clean data
    print("\n--- Data Cleaning ---")
    df_original = df.copy()
    
    # Remove rows with NaN in critical columns
    critical_cols = ['error_reduction', 'noisy_error', 'zne_error']
    before = len(df)
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} rows with NaN values")
    
    # Remove extreme outliers in error_reduction (if column exists)
    if 'error_reduction' in df.columns:
        print(f"  Original error_reduction range: [{df['error_reduction'].min():.4f}, {df['error_reduction'].max():.4f}]")
        extreme_low = (df['error_reduction'] < -2.0).sum()
        extreme_high = (df['error_reduction'] > 2.0).sum()
        print(f"    Extreme negative (<-2.0): {extreme_low} ({extreme_low/len(df)*100:.1f}%)")
        print(f"    Extreme positive (>2.0): {extreme_high} ({extreme_high/len(df)*100:.1f}%)")
        
        df = df[(df['error_reduction'] >= -2.0) & (df['error_reduction'] <= 2.0)].copy()
        print(f"  Retained: {len(df)}/{before} ({len(df)/before*100:.1f}%)")
    
    # Add derived metrics
    if 'noisy_error' in df.columns and 'zne_error' in df.columns:
        df['absolute_improvement'] = df['noisy_error'] - df['zne_error']
        df['error_ratio'] = df['zne_error'] / df['noisy_error'].replace(0, np.nan)
    
    if 'partitioned_depth' in df.columns and 'circuit_depth' in df.columns:
        df['depth_overhead'] = df['partitioned_depth'] / df['circuit_depth'].replace(0, np.nan)
    
    # Standardize communication primitive names
    if 'communication_primitive' in df.columns:
        df['comm_type'] = df['communication_primitive'].replace({'tp': 'teleportation', 'tg': 'teleportation'})
    
    # Extract algorithm family (first part before underscore)
    if 'origin' in df.columns:
        df['algorithm_family'] = df['origin'].str.extract(r'([a-z]+)_')[0]
    
    # Calculate network noise metric
    if 'num_partitions_tested' in df.columns and 'communication_noise_multiplier' in df.columns:
        df['network_noise_injected'] = df['num_partitions_tested'] * df['communication_noise_multiplier']
    
    print(f"\n✓ Data cleaned and augmented")
    print(f"  Dataset before partition filter: {len(df)} experiments")
    
    # Filter out partition=1 data (no distribution occurs)
    df = filter_distributed_data(df, keep_baseline=False)
    
    print(f"  Final dataset: {len(df)} experiments (partition > 1 only)")
    
    return df

def plot_baseline_validation(df_full, output_dir='figures'):
    """
    Create validation plot comparing partition=1 (baseline) vs partition>1.
    This helps verify that partition=1 shows no strategy differences (as expected).
    """
    print_section("BASELINE VALIDATION (Partition=1 Check)")
    
    if 'num_partitions_tested' not in df_full.columns:
        print("⚠ Skipping: 'num_partitions_tested' column not found")
        return
    
    df_baseline = df_full[df_full['num_partitions_tested'] == 1].copy()
    
    if len(df_baseline) == 0:
        print("⚠ No partition=1 data found, skipping baseline validation")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Check if strategies differ at partition=1 (they shouldn't!)
    print(f"\n--- Partition=1 Strategy Comparison ---")
    if 'strategy' in df_baseline.columns:
        baseline_stats = df_baseline.groupby('strategy')['error_reduction'].agg(['mean', 'std', 'count'])
        print(baseline_stats)
        
        # Calculate variance between strategies
        strategy_means = baseline_stats['mean'].values
        if len(strategy_means) > 1:
            variance = np.var(strategy_means)
            print(f"\nVariance between strategies: {variance:.6f}")
            if variance > 0.01:
                print("⚠ WARNING: Strategies differ significantly at partition=1!")
                print("   This suggests a bug - all strategies should be identical with no distribution.")
            else:
                print("✓ Good: Strategies are similar at partition=1 (as expected)")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Partition=1 by strategy
    if 'strategy' in df_baseline.columns:
        ax = axes[0]
        strategies = sorted(df_baseline['strategy'].unique())
        positions = np.arange(len(strategies))
        
        means = []
        stds = []
        for strategy in strategies:
            data = df_baseline[df_baseline['strategy'] == strategy]['error_reduction']
            means.append(data.mean())
            stds.append(data.std())
        
        ax.bar(positions, means, yerr=stds, alpha=0.7, capsize=5)
        ax.set_xticks(positions)
        ax.set_xticklabels([s.upper() for s in strategies])
        ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
        # ax.set_title('Partition=1: Strategy Comparison\n(Should be identical)', fontsize=12, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Distribution of error reduction at partition=1
    ax = axes[1]
    ax.hist(df_baseline['error_reduction'], bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(df_baseline['error_reduction'].mean(), color='red', 
              linestyle='--', linewidth=2, label=f"Mean: {df_baseline['error_reduction'].mean():.4f}")
    ax.set_xlabel('Error Reduction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    # ax.set_title('Partition=1: Error Reduction Distribution\n(No Distribution)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = Path(output_dir) / 'fig0_baseline_validation_partition1.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filepath}")
    plt.close()

def plot_scalability(df, output_dir='figures'):
    """Create scalability analysis plots"""
    print_section("GENERATING SCALABILITY PLOTS")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Figure 1a: Error Reduction vs Partitions
    fig, ax = plt.subplots(figsize=(8, 6))
    for strategy in sorted(df['strategy'].unique()):
        df_strat = df[df['strategy'] == strategy]
        grouped = df_strat.groupby('num_partitions_tested').agg({
            'error_reduction': ['mean', 'std', 'count']
        })
        
        x = grouped.index
        y = grouped['error_reduction']['mean']
        err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
        
        ax.plot(x, y, marker='o', linewidth=2.5, markersize=8, label=strategy.upper(), alpha=0.8)
        ax.fill_between(x, y-err, y+err, alpha=0.2)
    
    ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
    # ax.set_title('Scalability: Error Reduction vs Partitions', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    filepath = Path(output_dir) / 'fig1a_error_reduction_vs_partitions.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()
    
    # Figure 1b: ZNE Error vs Partitions
    fig, ax = plt.subplots(figsize=(8, 6))
    for strategy in sorted(df['strategy'].unique()):
        df_strat = df[df['strategy'] == strategy]
        grouped = df_strat.groupby('num_partitions_tested')['zne_error'].agg(['mean', 'std', 'count'])
        
        x = grouped.index
        y = grouped['mean']
        err = grouped['std'] / np.sqrt(grouped['count'])
        
        ax.plot(x, y, marker='s', linewidth=2.5, markersize=8, label=strategy.upper(), alpha=0.8)
        ax.fill_between(x, y-err, y+err, alpha=0.2)
    
    ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
    ax.set_ylabel('ZNE Error', fontsize=12, fontweight='bold')
    # ax.set_title('Absolute Error vs Partitions', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    filepath = Path(output_dir) / 'fig1b_absolute_error_vs_partitions.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")
    plt.close()
    
    # Figure 1c: Depth Overhead vs Partitions (if available)
    if 'depth_overhead' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for strategy in sorted(df['strategy'].unique()):
            df_strat = df[df['strategy'] == strategy]
            grouped = df_strat.groupby('num_partitions_tested')['depth_overhead'].mean()
            
            ax.plot(grouped.index, grouped.values, marker='^', linewidth=2.5, markersize=8, 
                   label=strategy.upper(), alpha=0.8)
        
        ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Depth Overhead (ratio)', fontsize=12, fontweight='bold')
        # ax.set_title('Circuit Depth Penalty', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig1c_depth_overhead_vs_partitions.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    # Figure 2a: Error Reduction vs Network Noise by Strategy
    if 'communication_noise_multiplier' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for strategy in sorted(df['strategy'].unique()):
            df_strat = df[df['strategy'] == strategy]
            grouped = df_strat.groupby('communication_noise_multiplier').agg({
                'error_reduction': ['mean', 'std', 'count']
            })
            
            x = grouped.index
            y = grouped['error_reduction']['mean']
            err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
            
            ax.plot(x, y, marker='o', linewidth=2.5, markersize=8, label=strategy.upper(), alpha=0.8)
            ax.fill_between(x, y-err, y+err, alpha=0.2)
        
        ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
        # ax.set_title('Network Noise Resistance', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig2a_network_noise_resistance.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
        
        # Custom aggregation function: trimmed mean (removes outliers)
        def robust_mean(x):
            """Calculate mean after removing outliers using IQR method"""
            if len(x) < 3:  # Need at least 3 points
                return x.mean()
            
            q1, q3 = np.percentile(x, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Filter outliers
            filtered = x[(x >= lower_bound) & (x <= upper_bound)]
            
            # Return mean of filtered data, or original mean if too few points remain
            return filtered.mean() if len(filtered) >= 2 else x.mean()
        
        # Create separate pivot tables for global and local strategies
        df_global = df[df['strategy'] == 'global']
        df_local = df[df['strategy'] == 'local']
        
        pivot_global = df_global.pivot_table(
            values='error_reduction',
            index='num_partitions_tested',
            columns='communication_noise_multiplier',
            aggfunc=robust_mean
        )
        
        pivot_local = df_local.pivot_table(
            values='error_reduction',
            index='num_partitions_tested',
            columns='communication_noise_multiplier',
            aggfunc=robust_mean
        )
        
        # Figure 2b: Global Strategy Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_global, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   ax=ax, cbar_kws={'label': 'Error Reduction'}, 
                   vmin=-0.1, vmax=0.2)
        ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Partitions', fontsize=12, fontweight='bold')
        # ax.set_title('Error Reduction Heatmap: Global ZNE', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig2b_error_reduction_heatmap_global.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
        
        # Figure 2c: Local Strategy Heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_local, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   ax=ax, cbar_kws={'label': 'Error Reduction'}, 
                   vmin=-0.1, vmax=0.2)
        ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Partitions', fontsize=12, fontweight='bold')
        # ax.set_title('Error Reduction Heatmap: Local ZNE', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig2c_error_reduction_heatmap_local.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()

    
    # Figure 3a: Error Reduction by Local Noise
    if 'local_noise' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for strategy in sorted(df['strategy'].unique()):
            df_strat = df[df['strategy'] == strategy]
            grouped = df_strat.groupby('local_noise').agg({
                'error_reduction': ['mean', 'std', 'count']
            })
            
            x = grouped.index
            y = grouped['error_reduction']['mean']
            err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
            
            ax.plot(x, y, marker='o', linewidth=2.5, markersize=8, label=strategy.upper(), alpha=0.8)
            ax.fill_between(x, y-err, y+err, alpha=0.2)
        
        ax.set_xlabel('Local Noise Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
        # ax.set_title('Performance vs Local Noise', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig3a_performance_vs_local_noise.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
        
        # Figure 3b: Box plot comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        df_plot = df[['strategy', 'error_reduction']].copy()
        sns.boxplot(data=df_plot, x='strategy', y='error_reduction', ax=ax, palette='Set2')
        ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
        # ax.set_title('Strategy Performance Distribution', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig3b_strategy_performance_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    # Figure 4a: Error reduction by algorithm family (overview)
    if 'algorithm_family' in df.columns:
        families = sorted(df['algorithm_family'].value_counts().head(3).index)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for family in families:
            df_fam = df[df['algorithm_family'] == family]
            grouped = df_fam.groupby('num_partitions_tested').agg({
                'error_reduction': ['mean', 'std', 'count']
            })
            
            x = grouped.index
            y = grouped['error_reduction']['mean']
            err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
            
            ax.plot(x, y, marker='o', linewidth=2.5, markersize=8, label=family.upper(), alpha=0.8)
            ax.fill_between(x, y-err, y+err, alpha=0.2)
        
        ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
        # ax.set_title('Performance by Algorithm Family', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig4a_performance_by_algorithm_family.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
        
        # Figures 4b-4d: Strategy comparison for each algorithm family
        for idx, family in enumerate(families):
            fig, ax = plt.subplots(figsize=(8, 6))
            df_fam = df[df['algorithm_family'] == family]
            
            for strategy in sorted(df_fam['strategy'].unique()):
                df_strat = df_fam[df_fam['strategy'] == strategy]
                grouped = df_strat.groupby('num_partitions_tested').agg({
                    'error_reduction': ['mean', 'std', 'count']
                })
                
                if len(grouped) > 0:
                    x = grouped.index
                    y = grouped['error_reduction']['mean']
                    err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
                    
                    ax.plot(x, y, marker='s', linewidth=2.5, markersize=8,
                           label=strategy.upper(), alpha=0.8)
                    ax.fill_between(x, y-err, y+err, alpha=0.2)
            
            ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
            ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
            # ax.set_title(f'Strategy Comparison: {family.upper()} Circuits', fontsize=14, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            plt.tight_layout()
            filepath = Path(output_dir) / f'fig4{chr(98+idx)}_strategy_comparison_{family}.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")
            plt.close()


def statistical_analysis(df):
    """Perform statistical analysis"""
    print_section("STATISTICAL ANALYSIS")
    
    # Overall summary
    print("\n--- Overall Performance Summary ---")
    summary = df.groupby('strategy').agg({
        'error_reduction': ['mean', 'std', 'count'],
        'zne_error': 'mean',
        'noisy_error': 'mean'
    }).round(4)
    print(summary)
    
    # Strategy comparison test
    if len(df['strategy'].unique()) >= 2:
        print("\n--- Strategy Comparison (Mann-Whitney U Test) ---")
        strategies = sorted(df['strategy'].unique())
        for i, strat1 in enumerate(strategies[:-1]):
            for strat2 in strategies[i+1:]:
                data1 = df[df['strategy'] == strat1]['error_reduction']
                data2 = df[df['strategy'] == strat2]['error_reduction']
                
                statistic, pvalue = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                print(f"\n  {strat1.upper()} vs {strat2.upper()}:")
                print(f"    U-statistic: {statistic:.2f}")
                print(f"    p-value: {pvalue:.4f}")
                print(f"    Significant at α=0.05: {'Yes' if pvalue < 0.05 else 'No'}")
                print(f"    Mean difference: {data1.mean() - data2.mean():.4f}")
    
    # Correlation analysis
    if 'num_partitions_tested' in df.columns:
        print("\n--- Correlation: Partitions vs Error Reduction ---")
        for strategy in df['strategy'].unique():
            df_strat = df[df['strategy'] == strategy]
            corr, pval = stats.spearmanr(df_strat['num_partitions_tested'], 
                                         df_strat['error_reduction'])
            print(f"  {strategy.upper()}: ρ={corr:.3f}, p={pval:.4f}")
    
    # Best configurations
    print("\n--- Best Configurations ---")
    for strategy in df['strategy'].unique():
        df_strat = df[df['strategy'] == strategy]
        best_idx = df_strat['error_reduction'].idxmax()
        best_row = df_strat.loc[best_idx]
        
        print(f"\n  {strategy.upper()}:")
        print(f"    Error Reduction: {best_row['error_reduction']:.4f}")
        if 'num_partitions_tested' in best_row:
            print(f"    Partitions: {best_row['num_partitions_tested']}")
        if 'local_noise' in best_row:
            print(f"    Local Noise: {best_row['local_noise']}")
        if 'communication_noise_multiplier' in best_row:
            print(f"    Comm Noise Multiplier: {best_row['communication_noise_multiplier']}")

def main():
    """Main analysis pipeline"""
    # Get input file
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../results.csv'
    
    # Load full dataset (including partition=1 for validation)
    print_section("LOADING DATA")
    try:
        df_full = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df_full)} experiments from {input_file}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("\nUsage: python analyze_results.py [path_to_results.csv]")
        sys.exit(1)
    
    # Create baseline validation plot (uses partition=1 data)
    plot_baseline_validation(df_full)
    
    # Load and clean data (this filters out partition=1)
    df = load_and_clean_data(input_file)
    
    # Run main analysis on distributed data only (partition > 1)
    plot_scalability(df)
    statistical_analysis(df)
    
    print_section("ANALYSIS COMPLETE")
    print("\nGenerated files in 'figures/' directory:")
    for fig in sorted(Path('figures').glob('*.png')):
        print(f"  - {fig.name}")
    print("\n✓ All done!")
    print("\nNote: Analysis excludes partition=1 data (no distribution occurs)")
    print("      See fig0_baseline_validation_partition1.png for partition=1 check")

if __name__ == "__main__":
    main()