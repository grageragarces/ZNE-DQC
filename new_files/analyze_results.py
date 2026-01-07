"""
ZNE Distributed Quantum Computing Analysis
Analyzes results from main_reduced.ipynb experiments

Usage: python quick_ploys.py ../results.csv
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
    print(f"  Final dataset: {len(df)} experiments")
    
    return df

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
    ax.set_title('Scalability: Error Reduction vs Partitions', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
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
    ax.set_title('Absolute Error vs Partitions', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
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
        ax.set_title('Circuit Depth Penalty', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        
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
        ax.set_title('Network Noise Resistance', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig2a_network_noise_resistance.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
        
        # Figure 2b: Heatmap of error reduction
        fig, ax = plt.subplots(figsize=(8, 6))
        pivot_data = df.pivot_table(
            values='error_reduction',
            index='num_partitions_tested',
            columns='communication_noise_multiplier',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   ax=ax, cbar_kws={'label': 'Error Reduction'})
        ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Partitions', fontsize=12, fontweight='bold')
        ax.set_title('Error Reduction Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filepath = Path(output_dir) / 'fig2b_error_reduction_heatmap.png'
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
        ax.set_title('Performance vs Local Noise', fontsize=14, fontweight='bold')
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
        ax.set_title('Strategy Performance Distribution', fontsize=14, fontweight='bold')
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
        ax.set_title('Performance by Algorithm Family', fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
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
            ax.set_title(f'Strategy Comparison: {family.upper()} Circuits', 
                        fontsize=14, fontweight='bold')
            ax.legend(loc='best', frameon=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            
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
    
    # Run analysis
    df = load_and_clean_data(input_file)
    plot_scalability(df)
    statistical_analysis(df)
    
    print_section("ANALYSIS COMPLETE")
    print("\nGenerated files in 'figures/' directory:")
    for fig in sorted(Path('figures').glob('*.png')):
        print(f"  - {fig.name}")
    print("\n✓ All done!")

if __name__ == "__main__":
    main()