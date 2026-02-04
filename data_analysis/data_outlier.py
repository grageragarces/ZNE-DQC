"""
Investigate WHY error_reduction has extreme outliers.

This helps identify:
1. Which experiments produce extreme values
2. Whether they represent real failures or calculation errors
3. Whether filtering them out is appropriate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def investigate_outliers(df):
    """
    Analyze extreme error_reduction values to understand their cause.
    """
    
    # Separate into categories
    extreme_negative = df[df['error_reduction'] < -2.0].copy()
    extreme_positive = df[df['error_reduction'] > 2.0].copy()
    normal = df[(df['error_reduction'] >= -2.0) & (df['error_reduction'] <= 2.0)].copy()
    
    print(f"\n--- Data Distribution ---")
    print(f"Total experiments: {len(df)}")
    print(f"Normal range [-2, 2]: {len(normal)} ({len(normal)/len(df)*100:.1f}%)")
    print(f"Extreme negative (<-2): {len(extreme_negative)} ({len(extreme_negative)/len(df)*100:.1f}%)")
    print(f"Extreme positive (>2): {len(extreme_positive)} ({len(extreme_positive)/len(df)*100:.1f}%)")
    
    # Analyze extreme negative cases
    if len(extreme_negative) > 0:
        print("\n" + "="*80)
        print("EXTREME NEGATIVE: ZNE made things MUCH worse")
        print("="*80)
        print("\n--- Statistical Summary ---")
        print(extreme_negative[['error_reduction', 'noisy_error', 'zne_error']].describe())
        
        print("\n--- Worst 10 Cases ---")
        worst = extreme_negative.nsmallest(10, 'error_reduction')
        for idx, row in worst.iterrows():
            print(f"\nCase {idx}:")
            print(f"  Algorithm: {row.get('origin', 'unknown')}")
            if 'num_partitions_tested' in row:
                print(f"  Partitions: {row['num_partitions_tested']}")
            if 'strategy' in row:
                print(f"  Strategy: {row['strategy']}")
            print(f"  Noisy error: {row['noisy_error']:.6f}")
            print(f"  ZNE error: {row['zne_error']:.6f}")
            print(f"  Error reduction: {row['error_reduction']:.2e}")
            print(f"  → ZNE made error {row['zne_error']/row['noisy_error']:.1f}x WORSE")
        
        # Check if near-zero baselines are the cause
        near_zero_baseline = extreme_negative[extreme_negative['noisy_error'] < 0.01]
        print(f"\n--- Near-Zero Baseline Analysis ---")
        print(f"Extreme negatives with noisy_error < 0.01: {len(near_zero_baseline)}")
        if len(near_zero_baseline) > 0:
            print(f"  Mean noisy_error: {near_zero_baseline['noisy_error'].mean():.6f}")
            print(f"  Mean zne_error: {near_zero_baseline['zne_error'].mean():.6f}")
            print("\n  ⚠️  DIAGNOSIS: Division by near-zero baseline causes extreme values!")
            print("     When baseline is nearly perfect, any increase becomes huge percentage.")
    
    # Analyze extreme positive cases
    if len(extreme_positive) > 0:
        print("\n" + "="*80)
        print("EXTREME POSITIVE: ZNE worked UNREALISTICALLY well")
        print("="*80)
        print("\n--- Statistical Summary ---")
        print(extreme_positive[['error_reduction', 'noisy_error', 'zne_error']].describe())
        
        print("\n--- Best 10 Cases ---")
        best = extreme_positive.nlargest(10, 'error_reduction')
        for idx, row in best.iterrows():
            print(f"\nCase {idx}:")
            print(f"  Algorithm: {row.get('origin', 'unknown')}")
            if 'num_partitions_tested' in row:
                print(f"  Partitions: {row['num_partitions_tested']}")
            if 'strategy' in row:
                print(f"  Strategy: {row['strategy']}")
            print(f"  Noisy error: {row['noisy_error']:.6f}")
            print(f"  ZNE error: {row['zne_error']:.6f}")
            print(f"  Error reduction: {row['error_reduction']:.2e}")
            print(f"  → ZNE reduced error by {(1 - row['zne_error']/row['noisy_error'])*100:.1f}%")
        
        # Check for negative ZNE errors (impossible!)
        negative_zne = extreme_positive[extreme_positive['zne_error'] < 0]
        if len(negative_zne) > 0:
            print(f"\n  ⚠️  WARNING: {len(negative_zne)} cases with NEGATIVE zne_error!")
            print("     This is physically impossible and indicates a simulation bug.")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution of error_reduction
    ax = axes[0, 0]
    ax.hist(df['error_reduction'], bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(-2, color='red', linestyle='--', linewidth=2, label='Filter threshold')
    ax.axvline(2, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Error Reduction', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Full Distribution (with outliers)', fontweight='bold')
    ax.legend()
    ax.set_xlim(df['error_reduction'].quantile(0.001), df['error_reduction'].quantile(0.999))
    
    # 2. Filtered distribution
    ax = axes[0, 1]
    ax.hist(normal['error_reduction'], bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel('Error Reduction', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Filtered Distribution ([-2, 2] range)', fontweight='bold')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    # 3. Scatter: noisy_error vs zne_error (colored by error_reduction)
    ax = axes[1, 0]
    scatter = ax.scatter(df['noisy_error'], df['zne_error'], 
                        c=df['error_reduction'].clip(-2, 2), 
                        cmap='RdYlGn', alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x (no change)')
    ax.set_xlabel('Noisy Error (baseline)', fontweight='bold')
    ax.set_ylabel('ZNE Error (mitigated)', fontweight='bold')
    ax.set_title('Baseline vs Mitigated Error', fontweight='bold')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Error Reduction')
    ax.set_xlim(0, min(1, df['noisy_error'].quantile(0.99)))
    ax.set_ylim(0, min(1, df['zne_error'].quantile(0.99)))
    
    # 4. Box plot by category
    ax = axes[1, 1]
    data_to_plot = [
        normal['error_reduction'],
        extreme_negative['error_reduction'] if len(extreme_negative) > 0 else [],
        extreme_positive['error_reduction'] if len(extreme_positive) > 0 else []
    ]
    labels = ['Normal\n[-2, 2]', f'Extreme Neg\n({len(extreme_negative)})', f'Extreme Pos\n({len(extreme_positive)})']
    
    bp = ax.boxplot([d for d in data_to_plot if len(d) > 0], 
                     labels=[l for d, l in zip(data_to_plot, labels) if len(d) > 0],
                     patch_artist=True)
    ax.set_ylabel('Error Reduction', fontweight='bold')
    ax.set_title('Distribution by Category', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('figures/outlier_investigation.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: figures/outlier_investigation.png")
    plt.close()
    
def main():
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../results.csv'
    
    print(f"Loading data from: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} experiments")
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    
    # Filter to distributed data only
    if 'num_partitions_tested' in df.columns:
        df = df[df['num_partitions_tested'] > 1].copy()
        print(f"Using {len(df)} distributed experiments (partition > 1)")
    
    investigate_outliers(df)

if __name__ == "__main__":
    main()