"""
ZNE Scalability Visualizations

Creates comprehensive visualizations answering:
1. How does performance degrade with more partitions/cuts?
2. Which strategy is more resistant to network noise?
3. How do strategies compare at different scales?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("Generating Scalability Visualizations")
print("="*80)

# Load cleaned data
try:
    df = pd.read_csv('results_cleaned.csv')
    print(f"✓ Loaded {len(df)} cleaned experiments\n")
except:
    print("✗ Error: results_cleaned.csv not found. Run scalability_analysis.py first.\n")
    exit(1)

import os
os.makedirs('figures', exist_ok=True)

# ============================================================================
# FIGURE 1: Error vs Partitions by Strategy
# ============================================================================
print("Creating Figure 1: Error Reduction vs Number of Partitions...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1a: Error Reduction vs Partitions
ax = axes[0]
for strategy in sorted(df['strategy'].unique()):
    df_strat = df[df['strategy'] == strategy]
    grouped = df_strat.groupby('num_partitions_tested').agg({
        'error_reduction': ['mean', 'std', 'count']
    })
    
    x = grouped.index
    y = grouped['error_reduction']['mean']
    err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
    
    ax.plot(x, y, marker='o', linewidth=2, markersize=8, label=strategy.upper())
    ax.fill_between(x, y-err, y+err, alpha=0.2)

ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
ax.set_title('Scalability: Error Reduction vs Cuts', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Plot 1b: ZNE Error vs Partitions
ax = axes[1]
for strategy in sorted(df['strategy'].unique()):
    df_strat = df[df['strategy'] == strategy]
    grouped = df_strat.groupby('num_partitions_tested')['zne_error'].agg(['mean', 'std', 'count'])
    
    x = grouped.index
    y = grouped['mean']
    err = grouped['std'] / np.sqrt(grouped['count'])
    
    ax.plot(x, y, marker='s', linewidth=2, markersize=8, label=strategy.upper())
    ax.fill_between(x, y-err, y+err, alpha=0.2)

ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
ax.set_ylabel('ZNE Error', fontsize=12, fontweight='bold')
ax.set_title('Absolute Error vs Cuts', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)

# Plot 1c: Depth Overhead vs Partitions
ax = axes[2]
for strategy in sorted(df['strategy'].unique()):
    df_strat = df[df['strategy'] == strategy]
    grouped = df_strat.groupby('num_partitions_tested')['depth_overhead'].mean()
    
    ax.plot(grouped.index, grouped.values, marker='^', linewidth=2, markersize=8, label=strategy.upper())

ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
ax.set_ylabel('Depth Overhead (ratio)', fontsize=12, fontweight='bold')
ax.set_title('Circuit Depth Penalty', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('figures/fig1_partition_scaling.png', bbox_inches='tight')
print("  ✓ Saved: figures/fig1_partition_scaling.png")
plt.close()

# ============================================================================
# FIGURE 2: Network Noise Resistance
# ============================================================================
print("Creating Figure 2: Network Noise Resistance...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 2a: Error Reduction vs Communication Noise Multiplier
ax = axes[0, 0]
for strategy in sorted(df['strategy'].unique()):
    df_strat = df[df['strategy'] == strategy]
    grouped = df_strat.groupby('communication_noise_multiplier').agg({
        'error_reduction': ['mean', 'std', 'count']
    })
    
    x = grouped.index
    y = grouped['error_reduction']['mean']
    err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
    
    ax.plot(x, y, marker='o', linewidth=2.5, markersize=10, label=strategy.upper())
    ax.fill_between(x, y-err, y+err, alpha=0.2)

ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
ax.set_title('Network Noise Impact on Performance', fontsize=13, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 2b: Heatmap - Strategy × Noise Multiplier
ax = axes[0, 1]
pivot = df.pivot_table(
    values='error_reduction',
    index='strategy',
    columns='communication_noise_multiplier',
    aggfunc='median'
)
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
            cbar_kws={'label': 'Median Error Reduction'})
ax.set_title('Performance Heatmap:\nStrategy × Network Noise', fontsize=13, fontweight='bold')
ax.set_xlabel('Communication Noise Multiplier', fontsize=11, fontweight='bold')
ax.set_ylabel('Strategy', fontsize=11, fontweight='bold')

# Plot 2c: Total Network Noise Injected vs Error Reduction
ax = axes[1, 0]
for strategy in sorted(df['strategy'].unique()):
    df_strat = df[df['strategy'] == strategy]
    
    # Bin the data for clearer visualization
    bins = np.linspace(df_strat['network_noise_injected'].min(), 
                      df_strat['network_noise_injected'].max(), 10)
    df_strat['noise_bin'] = pd.cut(df_strat['network_noise_injected'], bins=bins)
    
    grouped = df_strat.groupby('noise_bin')['error_reduction'].agg(['mean', 'count'])
    grouped = grouped[grouped['count'] >= 5]  # Filter sparse bins
    
    if len(grouped) > 0:
        x = [interval.mid for interval in grouped.index]
        y = grouped['mean'].values
        
        ax.scatter(x, y, s=100, alpha=0.6, label=strategy.upper())
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), linestyle='--', alpha=0.5, linewidth=2)

ax.set_xlabel('Total Network Noise Injected\n(Partitions × Noise Multiplier)', fontsize=11, fontweight='bold')
ax.set_ylabel('Error Reduction', fontsize=11, fontweight='bold')
ax.set_title('Cumulative Network Noise Effect', fontsize=13, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 2d: Strategy Comparison by Noise Level
ax = axes[1, 1]
noise_levels = sorted(df['communication_noise_multiplier'].unique())
strategies = sorted(df['strategy'].unique())

x = np.arange(len(noise_levels))
width = 0.25

for i, strategy in enumerate(strategies):
    means = []
    for noise in noise_levels:
        subset = df[(df['strategy'] == strategy) & 
                   (df['communication_noise_multiplier'] == noise)]
        means.append(subset['error_reduction'].median())
    
    ax.bar(x + i*width, means, width, label=strategy.upper(), alpha=0.8)

ax.set_xlabel('Communication Noise Multiplier', fontsize=11, fontweight='bold')
ax.set_ylabel('Median Error Reduction', fontsize=11, fontweight='bold')
ax.set_title('Strategy Comparison\nAcross Noise Levels', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(noise_levels)
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figures/fig2_network_noise_resistance.png', bbox_inches='tight')
print("  ✓ Saved: figures/fig2_network_noise_resistance.png")
plt.close()

# ============================================================================
# FIGURE 3: Algorithm Family-Specific Scalability
# ============================================================================
print("Creating Figure 3: Algorithm Family-Specific Analysis...")

families = ['dj', 'ghz', 'wstate']
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, family in enumerate(families):
    ax = axes[idx]
    df_family = df[df['algorithm_family'] == family]
    
    if len(df_family) > 20:
        for strategy in sorted(df_family['strategy'].unique()):
            df_strat = df_family[df_family['strategy'] == strategy]
            
            if len(df_strat) > 5:
                grouped = df_strat.groupby('num_partitions_tested').agg({
                    'error_reduction': ['mean', 'std', 'count']
                })
                
                x = grouped.index
                y = grouped['error_reduction']['mean']
                err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
                
                ax.plot(x, y, marker='o', linewidth=2, markersize=8, label=strategy.upper())
                ax.fill_between(x, y-err, y+err, alpha=0.2)
        
        ax.set_xlabel('Number of Partitions', fontsize=11, fontweight='bold')
        ax.set_ylabel('Error Reduction', fontsize=11, fontweight='bold')
        ax.set_title(f'{family.upper()} Circuits', fontsize=13, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    else:
        ax.text(0.5, 0.5, f'Insufficient {family.upper()} data', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')

plt.tight_layout()
plt.savefig('figures/fig3_algorithm_family_scalability.png', bbox_inches='tight')
print("  ✓ Saved: figures/fig3_algorithm_family_scalability.png")
plt.close()

# ============================================================================
# FIGURE 4: Communication Protocol Comparison
# ============================================================================
print("Creating Figure 4: Communication Protocol Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 4a: Protocol comparison across partitions
ax = axes[0, 0]
for comm in sorted(df['comm_type'].unique()):
    df_comm = df[df['comm_type'] == comm]
    grouped = df_comm.groupby('num_partitions_tested').agg({
        'error_reduction': ['mean', 'std', 'count']
    })
    
    x = grouped.index
    y = grouped['error_reduction']['mean']
    err = grouped['error_reduction']['std'] / np.sqrt(grouped['error_reduction']['count'])
    
    ax.plot(x, y, marker='o', linewidth=2.5, markersize=10, label=comm.upper())
    ax.fill_between(x, y-err, y+err, alpha=0.2)

ax.set_xlabel('Number of Partitions', fontsize=12, fontweight='bold')
ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
ax.set_title('Protocol Performance vs Scale', fontsize=13, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 4b: Protocol × Strategy interaction
ax = axes[0, 1]
interaction = df.pivot_table(
    values='error_reduction',
    index='strategy',
    columns='comm_type',
    aggfunc='median'
)
sns.heatmap(interaction, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
            cbar_kws={'label': 'Median Error Reduction'})
ax.set_title('Strategy × Protocol Interaction', fontsize=13, fontweight='bold')
ax.set_xlabel('Communication Protocol', fontsize=11, fontweight='bold')
ax.set_ylabel('Strategy', fontsize=11, fontweight='bold')

# Plot 4c: Protocol comparison by noise multiplier
ax = axes[1, 0]
for comm in sorted(df['comm_type'].unique()):
    df_comm = df[df['comm_type'] == comm]
    grouped = df_comm.groupby('communication_noise_multiplier')['error_reduction'].agg(['mean', 'std', 'count'])
    
    x = grouped.index
    y = grouped['mean']
    err = grouped['std'] / np.sqrt(grouped['count'])
    
    ax.plot(x, y, marker='s', linewidth=2.5, markersize=10, label=comm.upper())
    ax.fill_between(x, y-err, y+err, alpha=0.2)

ax.set_xlabel('Communication Noise Multiplier', fontsize=12, fontweight='bold')
ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
ax.set_title('Protocol Noise Resistance', fontsize=13, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# Plot 4d: Distribution comparison
ax = axes[1, 1]
df_plot = df[['comm_type', 'error_reduction']].copy()
df_plot['comm_type'] = df_plot['comm_type'].str.upper()

bp = ax.boxplot([df[df['comm_type'] == comm]['error_reduction'].values 
                 for comm in sorted(df['comm_type'].unique())],
                labels=[comm.upper() for comm in sorted(df['comm_type'].unique())],
                patch_artist=True, showmeans=True)

for patch, color in zip(bp['boxes'], sns.color_palette('husl', 2)):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('Error Reduction', fontsize=12, fontweight='bold')
ax.set_title('Protocol Performance Distribution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figures/fig4_communication_protocol.png', bbox_inches='tight')
print("  ✓ Saved: figures/fig4_communication_protocol.png")
plt.close()

# ============================================================================
# FIGURE 5: Comprehensive Summary Dashboard
# ============================================================================
print("Creating Figure 5: Summary Dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 5a: Overall strategy performance
ax = fig.add_subplot(gs[0, :2])
strategy_summary = df.groupby('strategy').agg({
    'error_reduction': ['median', 'mean', 'std'],
    'origin': 'count'
})
strategy_summary.columns = ['Median', 'Mean', 'Std', 'Count']

x = range(len(strategy_summary))
colors = sns.color_palette('husl', len(strategy_summary))

bars = ax.bar(x, strategy_summary['Median'], yerr=strategy_summary['Std'], 
              color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)

for i, (idx, row) in enumerate(strategy_summary.iterrows()):
    ax.text(i, row['Median'] + row['Std'] + 0.02, f"n={int(row['Count'])}", 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([idx.upper() for idx in strategy_summary.index], fontsize=11, fontweight='bold')
ax.set_ylabel('Median Error Reduction', fontsize=12, fontweight='bold')
ax.set_title('Overall Strategy Performance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)

# 5b: Scalability resistance scores
ax = fig.add_subplot(gs[0, 2])
resistance_scores = {}
for strategy in df['strategy'].unique():
    df_strat = df[df['strategy'] == strategy]
    corr, _ = spearmanr(df_strat['num_partitions_tested'], df_strat['error_reduction'])
    # Convert correlation to resistance score (lower correlation = better resistance)
    resistance_scores[strategy] = 1 - abs(corr)

strategies = list(resistance_scores.keys())
scores = list(resistance_scores.values())

bars = ax.barh(strategies, scores, color=sns.color_palette('RdYlGn', len(strategies)))
ax.set_xlabel('Resistance Score', fontsize=11, fontweight='bold')
ax.set_title('Scalability\nResistance', fontsize=12, fontweight='bold')
ax.set_xlim([0, 1.1])
for i, v in enumerate(scores):
    ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')

# 5c: Partition scaling trends
ax = fig.add_subplot(gs[1, :])
for strategy in sorted(df['strategy'].unique()):
    df_strat = df[df['strategy'] == strategy]
    grouped = df_strat.groupby('num_partitions_tested')['error_reduction'].median()
    
    ax.plot(grouped.index, grouped.values, marker='o', linewidth=2.5, 
            markersize=10, label=strategy.upper(), alpha=0.8)

ax.set_xlabel('Number of Partitions (Cuts)', fontsize=12, fontweight='bold')
ax.set_ylabel('Median Error Reduction', fontsize=12, fontweight='bold')
ax.set_title('Partition Scaling Trends', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)

# 5d: Network noise resistance
ax = fig.add_subplot(gs[2, 0])
noise_resistance = {}
for strategy in df['strategy'].unique():
    df_strat = df[df['strategy'] == strategy]
    corr, _ = spearmanr(df_strat['network_noise_injected'], df_strat['error_reduction'])
    noise_resistance[strategy] = 1 - abs(corr)

strategies = list(noise_resistance.keys())
scores = list(noise_resistance.values())

bars = ax.bar(strategies, scores, color=sns.color_palette('RdYlGn', len(strategies)), alpha=0.7)
ax.set_ylabel('Resistance Score', fontsize=11, fontweight='bold')
ax.set_title('Network Noise\nResistance', fontsize=12, fontweight='bold')
ax.set_ylim([0, 1.1])
for i, v in enumerate(scores):
    ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')

# 5e: Protocol comparison
ax = fig.add_subplot(gs[2, 1])
protocol_summary = df.groupby('comm_type')['error_reduction'].agg(['median', 'count'])

bars = ax.bar(range(len(protocol_summary)), protocol_summary['median'], 
              color=sns.color_palette('Set2', len(protocol_summary)), alpha=0.7)

ax.set_xticks(range(len(protocol_summary)))
ax.set_xticklabels([idx.upper() for idx in protocol_summary.index], fontsize=10, fontweight='bold')
ax.set_ylabel('Median Error Reduction', fontsize=11, fontweight='bold')
ax.set_title('Protocol\nComparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

# 5f: Key metrics table
ax = fig.add_subplot(gs[2, 2])
ax.axis('off')

# Create summary table
table_data = []
for strategy in sorted(df['strategy'].unique()):
    df_strat = df[df['strategy'] == strategy]
    median_er = df_strat['error_reduction'].median()
    n_exp = len(df_strat)
    
    # Scalability
    corr_part, _ = spearmanr(df_strat['num_partitions_tested'], df_strat['error_reduction'])
    if abs(corr_part) < 0.1:
        scale = "★★★"
    elif abs(corr_part) < 0.2:
        scale = "★★"
    else:
        scale = "★"
    
    table_data.append([strategy.upper()[:4], f"{median_er:.3f}", scale, n_exp])

table = ax.table(cellText=table_data, 
                colLabels=['Strat', 'Median\nER', 'Scale', 'N'],
                cellLoc='center', loc='center',
                colWidths=[0.25, 0.25, 0.25, 0.25])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color code the headers
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax.set_title('Key Metrics', fontsize=12, fontweight='bold')

plt.savefig('figures/fig5_summary_dashboard.png', bbox_inches='tight')
print("  ✓ Saved: figures/fig5_summary_dashboard.png")
plt.close()

print("\n" + "="*80)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*80)
print("\nGenerated figures:")
print("  1. fig1_partition_scaling.png        - Error vs partition count")
print("  2. fig2_network_noise_resistance.png - Network noise impact")
print("  3. fig3_algorithm_family_scalability.png - Family-specific analysis")
print("  4. fig4_communication_protocol.png   - Protocol comparison")
print("  5. fig5_summary_dashboard.png        - Comprehensive overview")
print("\nAll figures saved in: ./figures/")
print("\n")
