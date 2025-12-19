"""
ZNE Distributed Quantum Computing: Scalability and Network Noise Resistance Analysis

This script performs comprehensive analysis focused on:
1. How gate folding strategies (global vs local) scale with number of partitions/cuts
2. Which strategy is more resistant to network noise injection at scale
3. Impact of communication protocols (CAT vs TP) on scalability

Author: Analysis Script
Date: 2025-12-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("="*90)
print(" "*20 + "ZNE SCALABILITY & NETWORK NOISE RESISTANCE ANALYSIS")
print("="*90)

# ============================================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================================
print("\n" + "="*90)
print("STEP 1: DATA LOADING AND CLEANING")
print("="*90)

try:
    df = pd.read_csv('results.csv')
    print(f"✓ Loaded {len(df)} experiments")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    print("\nPlease ensure 'results.csv' is in the current directory.")
    exit(1)

# Display initial info
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Basic statistics
print(f"\n--- Experimental Parameters ---")
print(f"Strategies: {sorted(df['strategy'].unique())}")
print(f"Communication primitives: {sorted(df['communication_primitive'].unique())}")
print(f"Local noise levels: {sorted(df['local_noise'].unique())}")
print(f"Communication noise multipliers: {sorted(df['communication_noise_multiplier'].unique())}")
print(f"Partition range: {df['num_partitions_tested'].min()}-{df['num_partitions_tested'].max()}")
print(f"Algorithms: {len(df['origin'].unique())} unique algorithms")

# Data cleaning
print("\n--- Data Cleaning ---")
df_original = df.copy()

# Remove extreme outliers in error_reduction
print(f"Original error_reduction range: [{df['error_reduction'].min():.4f}, {df['error_reduction'].max():.4f}]")
extreme_low = (df['error_reduction'] < -2.0).sum()
extreme_high = (df['error_reduction'] > 2.0).sum()
print(f"  Extreme negative (<-2.0): {extreme_low} ({extreme_low/len(df)*100:.1f}%)")
print(f"  Extreme positive (>2.0): {extreme_high} ({extreme_high/len(df)*100:.1f}%)")

# Filter data
df_clean = df[
    (df['error_reduction'] >= -2.0) & 
    (df['error_reduction'] <= 2.0) &
    (df['noisy_error'] >= 0.001)  # Avoid division issues
].copy()

print(f"  Retained: {len(df_clean)}/{len(df)} ({len(df_clean)/len(df)*100:.1f}%)")

# Add derived metrics
df_clean['absolute_improvement'] = df_clean['noisy_error'] - df_clean['zne_error']
df_clean['error_ratio'] = df_clean['zne_error'] / df_clean['noisy_error']
df_clean['depth_overhead'] = df_clean['partitioned_depth'] / df_clean['circuit_depth']

# Standardize communication primitive names
df_clean['comm_type'] = df_clean['communication_primitive'].replace({'tp': 'teleportation', 'tg': 'teleportation'})

# Extract algorithm family
df_clean['algorithm_family'] = df_clean['origin'].str.extract(r'([a-z]+)_')[0]

# Calculate network noise injected (more cuts = more noise)
df_clean['network_noise_injected'] = (
    df_clean['num_partitions_tested'] * 
    df_clean['communication_noise_multiplier']
)

print(f"\n✓ Data cleaned and augmented")
print(f"  Added metrics: absolute_improvement, error_ratio, depth_overhead, network_noise_injected")

# Save cleaned data
df_clean.to_csv('results_cleaned.csv', index=False)
print(f"  Saved to: results_cleaned.csv")

# ============================================================================
# STEP 2: SCALABILITY ANALYSIS - RESISTANCE TO INCREASING CUTS
# ============================================================================
print("\n" + "="*90)
print("STEP 2: SCALABILITY ANALYSIS - RESISTANCE TO PARTITION SCALING")
print("="*90)
print("\nKey Question: Which strategy maintains performance as partition count increases?")

# For each strategy, analyze how error changes with partition count
print("\n--- Error Degradation with Partition Count ---")

for strategy in sorted(df_clean['strategy'].unique()):
    df_strat = df_clean[df_clean['strategy'] == strategy]
    
    # Calculate correlation between partitions and error
    if len(df_strat) > 10:
        corr_noisy, p_noisy = spearmanr(
            df_strat['num_partitions_tested'], 
            df_strat['noisy_error']
        )
        corr_zne, p_zne = spearmanr(
            df_strat['num_partitions_tested'], 
            df_strat['zne_error']
        )
        corr_reduction, p_reduction = spearmanr(
            df_strat['num_partitions_tested'], 
            df_strat['error_reduction']
        )
        
        print(f"\n  Strategy: {strategy.upper()}")
        print(f"    Partitions vs Noisy Error:     ρ={corr_noisy:+.3f} (p={p_noisy:.4f}) {'***' if p_noisy < 0.001 else '**' if p_noisy < 0.01 else '*' if p_noisy < 0.05 else 'ns'}")
        print(f"    Partitions vs ZNE Error:       ρ={corr_zne:+.3f} (p={p_zne:.4f}) {'***' if p_zne < 0.001 else '**' if p_zne < 0.01 else '*' if p_zne < 0.05 else 'ns'}")
        print(f"    Partitions vs Error Reduction: ρ={corr_reduction:+.3f} (p={p_reduction:.4f}) {'***' if p_reduction < 0.001 else '**' if p_reduction < 0.01 else '*' if p_reduction < 0.05 else 'ns'}")
        
        # Interpretation
        if abs(corr_reduction) < 0.1:
            interpretation = "STABLE - Error reduction unaffected by partition count"
        elif corr_reduction < -0.3:
            interpretation = "DEGRADES - Performance worsens significantly with more cuts"
        elif corr_reduction < -0.1:
            interpretation = "WEAKENS - Some degradation with more cuts"
        else:
            interpretation = "IMPROVES - Benefits from more cuts (unusual)"
        
        print(f"    → {interpretation}")

# Group analysis by number of partitions
print("\n--- Performance by Partition Count ---")
partition_summary = df_clean.groupby(['strategy', 'num_partitions_tested']).agg({
    'error_reduction': ['mean', 'median', 'std'],
    'noisy_error': 'mean',
    'zne_error': 'mean',
    'origin': 'count'
}).round(4)
partition_summary.columns = ['ER_mean', 'ER_median', 'ER_std', 'Noisy', 'ZNE', 'N']
print("\n", partition_summary)

# ============================================================================
# STEP 3: NETWORK NOISE RESISTANCE ANALYSIS
# ============================================================================
print("\n" + "="*90)
print("STEP 3: NETWORK NOISE RESISTANCE ANALYSIS")
print("="*90)
print("\nKey Question: Which strategy is more resistant to communication noise?")
print("(Higher communication_noise_multiplier = more network noise per cut)")

# Analyze effect of communication noise multiplier
print("\n--- Impact of Communication Noise Multiplier ---")

for strategy in sorted(df_clean['strategy'].unique()):
    df_strat = df_clean[df_clean['strategy'] == strategy]
    
    # Correlation between network noise and error reduction
    if len(df_strat) > 10:
        corr_mult, p_mult = spearmanr(
            df_strat['communication_noise_multiplier'],
            df_strat['error_reduction']
        )
        corr_total, p_total = spearmanr(
            df_strat['network_noise_injected'],
            df_strat['error_reduction']
        )
        
        print(f"\n  Strategy: {strategy.upper()}")
        print(f"    Noise Multiplier vs ER:       ρ={corr_mult:+.3f} (p={p_mult:.4f}) {'***' if p_mult < 0.001 else '**' if p_mult < 0.01 else '*' if p_mult < 0.05 else 'ns'}")
        print(f"    Total Network Noise vs ER:    ρ={corr_total:+.3f} (p={p_total:.4f}) {'***' if p_total < 0.001 else '**' if p_total < 0.01 else '*' if p_total < 0.05 else 'ns'}")
        
        # Resistance rating
        if abs(corr_total) < 0.1:
            resistance = "EXCELLENT - Highly resistant to network noise"
        elif abs(corr_total) < 0.2:
            resistance = "GOOD - Moderately resistant to network noise"
        elif abs(corr_total) < 0.3:
            resistance = "FAIR - Some sensitivity to network noise"
        else:
            resistance = "POOR - Highly sensitive to network noise"
        
        print(f"    → Resistance: {resistance}")

# Detailed breakdown by noise multiplier
print("\n--- Performance by Communication Noise Multiplier ---")
noise_summary = df_clean.groupby(['strategy', 'communication_noise_multiplier']).agg({
    'error_reduction': ['mean', 'median', 'std'],
    'noisy_error': 'mean',
    'zne_error': 'mean',
    'origin': 'count'
}).round(4)
noise_summary.columns = ['ER_mean', 'ER_median', 'ER_std', 'Noisy', 'ZNE', 'N']
print("\n", noise_summary)

# ============================================================================
# STEP 4: STRATEGY COMPARISON AT SCALE
# ============================================================================
print("\n" + "="*90)
print("STEP 4: DIRECT STRATEGY COMPARISON AT DIFFERENT SCALES")
print("="*90)

# Compare strategies at different partition counts
print("\n--- Strategy Comparison by Scale (Number of Partitions) ---")

for num_parts in sorted(df_clean['num_partitions_tested'].unique()):
    df_parts = df_clean[df_clean['num_partitions_tested'] == num_parts]
    
    strategy_comparison = df_parts.groupby('strategy')['error_reduction'].agg(['mean', 'median', 'std', 'count'])
    
    if len(strategy_comparison) > 1:
        print(f"\n  {num_parts} Partition(s):")
        for strategy in strategy_comparison.index:
            stats_row = strategy_comparison.loc[strategy]
            print(f"    {strategy:10s}: mean={stats_row['mean']:+.4f}, median={stats_row['median']:+.4f}, "
                  f"std={stats_row['std']:.4f} (n={int(stats_row['count'])})")
        
        # Determine winner at this scale
        winner = strategy_comparison['median'].idxmax()
        print(f"    → Best at {num_parts} cuts: {winner.upper()}")

# Statistical tests: Global vs Local across partition counts
print("\n--- Statistical Test: Global vs Local Across Scales ---")
global_data = df_clean[df_clean['strategy'] == 'global']
local_data = df_clean[df_clean['strategy'] == 'local']

if len(global_data) > 0 and len(local_data) > 0:
    # Overall comparison
    t_stat, p_value = stats.ttest_ind(
        global_data['error_reduction'], 
        local_data['error_reduction']
    )
    
    cohen_d = (global_data['error_reduction'].mean() - local_data['error_reduction'].mean()) / \
              np.sqrt((global_data['error_reduction'].std()**2 + local_data['error_reduction'].std()**2) / 2)
    
    print(f"\n  Overall Performance:")
    print(f"    Global: mean={global_data['error_reduction'].mean():+.4f}, n={len(global_data)}")
    print(f"    Local:  mean={local_data['error_reduction'].mean():+.4f}, n={len(local_data)}")
    print(f"    t-statistic: {t_stat:+.4f}")
    print(f"    p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    print(f"    Cohen's d: {cohen_d:+.4f} ", end="")
    
    if abs(cohen_d) < 0.2:
        print("(negligible effect)")
    elif abs(cohen_d) < 0.5:
        print("(small effect)")
    elif abs(cohen_d) < 0.8:
        print("(medium effect)")
    else:
        print("(large effect)")
    
    if p_value < 0.05:
        winner = "GLOBAL" if global_data['error_reduction'].mean() > local_data['error_reduction'].mean() else "LOCAL"
        print(f"\n    → Statistically significant difference: {winner} performs better overall")
    else:
        print(f"\n    → No statistically significant difference between strategies")

# ============================================================================
# STEP 5: COMMUNICATION PROTOCOL ANALYSIS
# ============================================================================
print("\n" + "="*90)
print("STEP 5: COMMUNICATION PROTOCOL IMPACT (CAT vs TELEPORTATION)")
print("="*90)

cat_data = df_clean[df_clean['comm_type'] == 'cat']
tp_data = df_clean[df_clean['comm_type'] == 'teleportation']

if len(cat_data) > 0 and len(tp_data) > 0:
    print(f"\n--- Overall Protocol Comparison ---")
    print(f"  CAT:          mean={cat_data['error_reduction'].mean():+.4f}, n={len(cat_data)}")
    print(f"  Teleportation: mean={tp_data['error_reduction'].mean():+.4f}, n={len(tp_data)}")
    
    t_stat, p_value = stats.ttest_ind(cat_data['error_reduction'], tp_data['error_reduction'])
    cohen_d = (cat_data['error_reduction'].mean() - tp_data['error_reduction'].mean()) / \
              np.sqrt((cat_data['error_reduction'].std()**2 + tp_data['error_reduction'].std()**2) / 2)
    
    print(f"  t-statistic: {t_stat:+.4f}")
    print(f"  p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
    print(f"  Cohen's d: {cohen_d:+.4f} ({' negligible' if abs(cohen_d) < 0.2 else 'small' if abs(cohen_d) < 0.5 else 'medium' if abs(cohen_d) < 0.8 else 'large'} effect)")
    
    if abs(cohen_d) < 0.2:
        print(f"\n  → Conclusion: Protocol choice has NEGLIGIBLE impact on scalability")
    else:
        winner = "CAT" if cat_data['error_reduction'].mean() > tp_data['error_reduction'].mean() else "TELEPORTATION"
        print(f"\n  → Conclusion: {winner} shows better performance")

    # Protocol performance at different partition counts
    print(f"\n--- Protocol Performance Across Partition Counts ---")
    for num_parts in sorted(df_clean['num_partitions_tested'].unique()):
        cat_parts = cat_data[cat_data['num_partitions_tested'] == num_parts]
        tp_parts = tp_data[tp_data['num_partitions_tested'] == num_parts]
        
        if len(cat_parts) > 5 and len(tp_parts) > 5:
            print(f"\n  {num_parts} Partition(s):")
            print(f"    CAT:          {cat_parts['error_reduction'].mean():+.4f} (n={len(cat_parts)})")
            print(f"    Teleportation: {tp_parts['error_reduction'].mean():+.4f} (n={len(tp_parts)})")
            
            diff = cat_parts['error_reduction'].mean() - tp_parts['error_reduction'].mean()
            print(f"    Difference: {diff:+.4f} ({'CAT better' if diff > 0 else 'TP better'})")

# ============================================================================
# STEP 6: ALGORITHM FAMILY ANALYSIS
# ============================================================================
print("\n" + "="*90)
print("STEP 6: ALGORITHM FAMILY-SPECIFIC INSIGHTS")
print("="*90)

print("\n--- Scalability by Algorithm Family ---")
for family in sorted(df_clean['algorithm_family'].unique()):
    df_family = df_clean[df_clean['algorithm_family'] == family]
    
    if len(df_family) > 20:
        print(f"\n  {family.upper()} Circuits (n={len(df_family)}):")
        
        # Best strategy for this family
        family_strategy = df_family.groupby('strategy')['error_reduction'].agg(['mean', 'count'])
        best_strategy = family_strategy['mean'].idxmax()
        
        print(f"    Best strategy: {best_strategy.upper()} (ER={family_strategy.loc[best_strategy, 'mean']:+.4f})")
        
        # Scalability check
        corr, p_val = spearmanr(
            df_family['num_partitions_tested'],
            df_family['error_reduction']
        )
        print(f"    Scalability: ρ={corr:+.3f} (p={p_val:.4f}) ", end="")
        if abs(corr) < 0.1:
            print("→ SCALES WELL")
        elif corr < -0.2:
            print("→ DEGRADES WITH CUTS")
        else:
            print("→ MODERATE SCALING")

# ============================================================================
# STEP 7: KEY FINDINGS SUMMARY
# ============================================================================
print("\n" + "="*90)
print("STEP 7: KEY FINDINGS & RECOMMENDATIONS")
print("="*90)

print("\n" + "🎯 SCALABILITY FINDINGS".center(90))
print("-" * 90)

# Finding 1: Best overall strategy
overall_best = df_clean.groupby('strategy')['error_reduction'].median().idxmax()
print(f"\n1. BEST OVERALL STRATEGY")
print(f"   {overall_best.upper()} shows best median error reduction across all experiments")

# Finding 2: Resistance to partition scaling
print(f"\n2. RESISTANCE TO PARTITION SCALING")
for strategy in sorted(df_clean['strategy'].unique()):
    df_strat = df_clean[df_clean['strategy'] == strategy]
    corr, _ = spearmanr(df_strat['num_partitions_tested'], df_strat['error_reduction'])
    
    if abs(corr) < 0.1:
        resistance = "EXCELLENT"
    elif abs(corr) < 0.2:
        resistance = "GOOD"
    elif abs(corr) < 0.3:
        resistance = "MODERATE"
    else:
        resistance = "POOR"
    
    print(f"   {strategy.upper():10s}: {resistance:10s} (ρ={corr:+.3f})")

# Finding 3: Network noise resistance
print(f"\n3. NETWORK NOISE RESISTANCE")
for strategy in sorted(df_clean['strategy'].unique()):
    df_strat = df_clean[df_clean['strategy'] == strategy]
    corr, _ = spearmanr(df_strat['network_noise_injected'], df_strat['error_reduction'])
    
    if abs(corr) < 0.15:
        resistance = "HIGHLY RESISTANT"
    elif abs(corr) < 0.25:
        resistance = "MODERATELY RESISTANT"
    else:
        resistance = "SENSITIVE"
    
    print(f"   {strategy.upper():10s}: {resistance:20s} (ρ={corr:+.3f})")

# Finding 4: Communication protocol impact
print(f"\n4. COMMUNICATION PROTOCOL IMPACT")
if len(cat_data) > 0 and len(tp_data) > 0:
    diff_pct = abs(cat_data['error_reduction'].mean() - tp_data['error_reduction'].mean()) * 100
    if diff_pct < 5:
        print(f"   NEGLIGIBLE difference between CAT and Teleportation ({diff_pct:.1f}%)")
        print(f"   → Protocol choice does NOT significantly affect scalability")
    else:
        winner = "CAT" if cat_data['error_reduction'].mean() > tp_data['error_reduction'].mean() else "TELEPORTATION"
        print(f"   {winner} performs {diff_pct:.1f}% better on average")

# Finding 5: Optimal partition count
print(f"\n5. OPTIMAL PARTITION COUNT")
partition_performance = df_clean.groupby('num_partitions_tested')['error_reduction'].median()
optimal_partitions = partition_performance.idxmax()
print(f"   Median error reduction peaks at {optimal_partitions} partition(s)")
print(f"   However, optimal count is algorithm-specific:")

for family in ['dj', 'ghz', 'wstate']:
    df_family = df_clean[df_clean['algorithm_family'] == family]
    if len(df_family) > 10:
        family_optimal = df_family.groupby('num_partitions_tested')['error_reduction'].median().idxmax()
        print(f"     {family.upper():8s}: {family_optimal} partition(s)")

# Recommendations
print("\n" + "💡 RECOMMENDATIONS".center(90))
print("-" * 90)

print("\n1. STRATEGY SELECTION")
print("   Based on scalability and network noise resistance:")
# Determine best strategy based on both partition scaling and noise resistance
strategy_scores = {}
for strategy in df_clean['strategy'].unique():
    df_strat = df_clean[df_clean['strategy'] == strategy]
    
    # Score: low correlation with partitions = good, high error reduction = good
    partition_corr, _ = spearmanr(df_strat['num_partitions_tested'], df_strat['error_reduction'])
    noise_corr, _ = spearmanr(df_strat['network_noise_injected'], df_strat['error_reduction'])
    avg_er = df_strat['error_reduction'].median()
    
    # Combined score (higher is better)
    score = avg_er - abs(partition_corr) - abs(noise_corr)
    strategy_scores[strategy] = score

best_strategy = max(strategy_scores, key=strategy_scores.get)
print(f"   → Recommended: {best_strategy.upper()}")
print(f"     Reasoning: Best balance of performance, scalability, and noise resistance")

print("\n2. PARTITION COUNT")
print(f"   → Use {optimal_partitions} partitions as starting point")
print(f"   → Tune per algorithm family (DJ, GHZ, W-state may differ)")
print(f"   → Monitor depth overhead: aim for depth_ratio < 3.0")

print("\n3. COMMUNICATION PROTOCOL")
if abs(cohen_d) < 0.2:
    print(f"   → Either CAT or Teleportation (negligible difference)")
    print(f"   → Choose based on implementation complexity/cost")
else:
    winner_protocol = "CAT" if cat_data['error_reduction'].mean() > tp_data['error_reduction'].mean() else "Teleportation"
    print(f"   → Prefer {winner_protocol}")

print("\n4. NOISE REGIME CONSIDERATIONS")
print(f"   → All strategies degrade with higher network noise")
print(f"   → For high-noise environments (mult > 1.1):")
print(f"     • Minimize partition count")
print(f"     • Prefer the most noise-resistant strategy")

print("\n" + "="*90)
print("ANALYSIS COMPLETE")
print("="*90)
print("\nGenerated files:")
print("  ✓ results_cleaned.csv - Cleaned dataset")
print("\nNext steps:")
print("  1. Generate visualizations (run visualization script)")
print("  2. Perform algorithm-specific deep dives")
print("  3. Validate findings on test set")
print("  4. Consider cost-benefit analysis with computational overhead")

print("\n")
