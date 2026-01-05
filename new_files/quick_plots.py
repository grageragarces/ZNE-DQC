"""
Quick Plot Generator for ZNE Analysis
Creates essential plots quickly for initial data exploration

Usage: python quick_plots.py [path_to_results.csv]
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.dpi'] = 150

def main():
    # Load data
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = '../results.csv'
    
    print(f"Loading {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} experiments")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Clean outliers
    df = df[(df['error_reduction'] >= -2.0) & (df['error_reduction'] <= 2.0)]
    print(f"✓ Cleaned to {len(df)} experiments (removed outliers)")
    
    # Create output directory
    Path('quick_plots').mkdir(exist_ok=True)
    
    # Plot 1: Error Reduction vs Partitions (by strategy)
    print("\n1. Plotting: Error Reduction vs Partitions...")
    plt.figure(figsize=(10, 6))
    
    for strategy in sorted(df['strategy'].unique()):
        df_strat = df[df['strategy'] == strategy]
        grouped = df_strat.groupby('num_partitions_tested')['error_reduction'].mean()
        plt.plot(grouped.index, grouped.values, marker='o', linewidth=2, 
                markersize=8, label=strategy.upper(), alpha=0.8)
    
    plt.xlabel('Number of Partitions', fontsize=12, fontweight='bold')
    plt.ylabel('Error Reduction', fontsize=12, fontweight='bold')
    plt.title('Error Reduction vs Number of Partitions', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('quick_plots/1_error_vs_partitions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: quick_plots/1_error_vs_partitions.png")
    
    # Plot 2: Box plot comparison
    print("2. Plotting: Strategy comparison boxplot...")
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(data=df, x='strategy', y='error_reduction', palette='Set2')
    plt.xlabel('Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Error Reduction', fontsize=12, fontweight='bold')
    plt.title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('quick_plots/2_strategy_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: quick_plots/2_strategy_boxplot.png")
    
    # Plot 3: ZNE Error vs Partitions
    print("3. Plotting: ZNE Error vs Partitions...")
    plt.figure(figsize=(10, 6))
    
    for strategy in sorted(df['strategy'].unique()):
        df_strat = df[df['strategy'] == strategy]
        grouped = df_strat.groupby('num_partitions_tested')['zne_error'].mean()
        plt.plot(grouped.index, grouped.values, marker='s', linewidth=2,
                markersize=8, label=strategy.upper(), alpha=0.8)
    
    plt.xlabel('Number of Partitions', fontsize=12, fontweight='bold')
    plt.ylabel('ZNE Error', fontsize=12, fontweight='bold')
    plt.title('Absolute ZNE Error vs Partitions', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('quick_plots/3_zne_error_vs_partitions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: quick_plots/3_zne_error_vs_partitions.png")
    
    # Plot 4: Communication noise effect
    if 'communication_noise_multiplier' in df.columns:
        print("4. Plotting: Communication noise effect...")
        plt.figure(figsize=(10, 6))
        
        for strategy in sorted(df['strategy'].unique()):
            df_strat = df[df['strategy'] == strategy]
            grouped = df_strat.groupby('communication_noise_multiplier')['error_reduction'].mean()
            plt.plot(grouped.index, grouped.values, marker='o', linewidth=2,
                    markersize=8, label=strategy.upper(), alpha=0.8)
        
        plt.xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
        plt.ylabel('Error Reduction', fontsize=12, fontweight='bold')
        plt.title('Network Noise Resistance', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('quick_plots/4_comm_noise_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: quick_plots/4_comm_noise_effect.png")
    
    # Plot 5: Local noise effect
    if 'local_noise' in df.columns:
        print("5. Plotting: Local noise effect...")
        plt.figure(figsize=(10, 6))
        
        for strategy in sorted(df['strategy'].unique()):
            df_strat = df[df['strategy'] == strategy]
            grouped = df_strat.groupby('local_noise')['error_reduction'].mean()
            plt.plot(grouped.index, grouped.values, marker='o', linewidth=2,
                    markersize=8, label=strategy.upper(), alpha=0.8)
        
        plt.xlabel('Local Noise Level', fontsize=12, fontweight='bold')
        plt.ylabel('Error Reduction', fontsize=12, fontweight='bold')
        plt.title('Performance vs Local Noise', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('quick_plots/5_local_noise_effect.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved: quick_plots/5_local_noise_effect.png")
    
    # Print summary stats
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for strategy in sorted(df['strategy'].unique()):
        df_strat = df[df['strategy'] == strategy]
        print(f"\n{strategy.upper()}:")
        print(f"  Mean Error Reduction: {df_strat['error_reduction'].mean():.4f}")
        print(f"  Std Error Reduction:  {df_strat['error_reduction'].std():.4f}")
        print(f"  Median Error Reduction: {df_strat['error_reduction'].median():.4f}")
        print(f"  Number of experiments: {len(df_strat)}")
    
    print("\n" + "="*60)
    print("✓ All plots saved to 'quick_plots/' directory")
    print("="*60)

if __name__ == "__main__":
    main()
