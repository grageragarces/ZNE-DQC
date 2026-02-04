"""
Investigate WHY some experiments have noisy_error ≈ 0 (perfect baseline).
This shouldn't happen in a noisy simulation.
"""

import pandas as pd
import numpy as np

def investigate_zero_baseline(df):
    """
    Find experiments with suspiciously low baseline error.
    """
    
    # Filter to distributed only
    if 'num_partitions_tested' in df.columns:
        df = df[df['num_partitions_tested'] > 1].copy()
    
    # Categorize by baseline error
    perfect = df[df['noisy_error'] == 0].copy()
    near_zero = df[(df['noisy_error'] > 0) & (df['noisy_error'] < 0.01)].copy()
    normal = df[df['noisy_error'] >= 0.01].copy()
    
    print(f"\n--- Baseline Error Distribution ---")
    print(f"Perfect (noisy_error = 0.0):     {len(perfect):4d} ({len(perfect)/len(df)*100:5.1f}%)")
    print(f"Near-zero (0 < noisy_error < 0.01): {len(near_zero):4d} ({len(near_zero)/len(df)*100:5.1f}%)")
    print(f"Normal (noisy_error >= 0.01):    {len(normal):4d} ({len(normal)/len(df)*100:5.1f}%)")
    
    if len(perfect) > 0:
        print("\nThese are SUSPICIOUS - noisy simulations shouldn't produce zero error!")
        
        # Analyze by algorithm
        print("\n--- By Algorithm ---")
        algo_counts = perfect['origin'].value_counts()
        for algo, count in algo_counts.head(10).items():
            pct = count / len(perfect) * 100
            print(f"  {algo:20s}: {count:3d} ({pct:5.1f}%)")
        
        # Analyze by partition count
        if 'num_partitions_tested' in perfect.columns:
            print("\n--- By Partition Count ---")
            part_counts = perfect['num_partitions_tested'].value_counts().sort_index()
            for part, count in part_counts.items():
                pct = count / len(perfect) * 100
                print(f"  {part} partitions: {count:3d} ({pct:5.1f}%)")
        
        # Analyze by strategy
        if 'strategy' in perfect.columns:
            print("\n--- By Strategy ---")
            strat_counts = perfect['strategy'].value_counts()
            for strat, count in strat_counts.items():
                pct = count / len(perfect) * 100
                print(f"  {strat:10s}: {count:3d} ({pct:5.1f}%)")
        
        # Analyze by noise level
        if 'local_noise' in perfect.columns:
            print("\n--- By Local Noise Level ---")
            noise_counts = perfect['local_noise'].value_counts().sort_index()
            for noise, count in noise_counts.items():
                pct = count / len(perfect) * 100
                print(f"  {noise:.3f}: {count:3d} ({pct:5.1f}%)")
        
        # Check if these are small circuits
        if 'circuit_qubits' in perfect.columns:
            print("\n--- By Circuit Size ---")
            print(f"  Mean qubits (perfect):  {perfect['circuit_qubits'].mean():.1f}")
            print(f"  Mean qubits (normal):   {normal['circuit_qubits'].mean():.1f}")
            print(f"  Mean qubits (all):      {df['circuit_qubits'].mean():.1f}")
        
        # Sample some cases
        print("\n--- Sample Perfect Baseline Cases ---")
        sample = perfect.head(5)
        for idx, row in sample.iterrows():
            print(f"\nCase {idx}:")
            print(f"  Algorithm: {row.get('origin', 'unknown')}")
            print(f"  Qubits: {row.get('circuit_qubits', 'unknown')}")
            if 'num_partitions_tested' in row:
                print(f"  Partitions: {row['num_partitions_tested']}")
            if 'strategy' in row:
                print(f"  Strategy: {row['strategy']}")
            if 'local_noise' in row:
                print(f"  Local noise: {row['local_noise']}")
            print(f"  Noisy error: {row['noisy_error']:.6f} (should NOT be zero!)")
            print(f"  ZNE error: {row['zne_error']:.6f}")
            if 'circuit_depth' in row:
                print(f"  Circuit depth: {row['circuit_depth']}")
    
    # Hypothesis testing

    # H1: Are perfect baselines more common in certain algorithms?
    if len(perfect) > 0 and 'origin' in df.columns:
        print("\nH1: Are certain algorithms more likely to have zero baseline?")
        for algo in df['origin'].unique()[:5]:  # Check top 5 algorithms
            algo_df = df[df['origin'] == algo]
            perfect_rate = (algo_df['noisy_error'] == 0).sum() / len(algo_df) * 100
            print(f"  {algo:20s}: {perfect_rate:5.1f}% have zero baseline")
    
    # H2: Are perfect baselines more common with fewer partitions?
    if len(perfect) > 0 and 'num_partitions_tested' in df.columns:
        print("\nH2: Do fewer partitions correlate with zero baseline?")
        for part in sorted(df['num_partitions_tested'].unique()):
            part_df = df[df['num_partitions_tested'] == part]
            perfect_rate = (part_df['noisy_error'] == 0).sum() / len(part_df) * 100
            print(f"  {part} partitions: {perfect_rate:5.1f}% have zero baseline")
    
    # H3: Are perfect baselines more common with lower noise?
    if len(perfect) > 0 and 'local_noise' in df.columns:
        print("\nH3: Does lower noise correlate with zero baseline?")
        for noise in sorted(df['local_noise'].unique()):
            noise_df = df[df['local_noise'] == noise]
            perfect_rate = (noise_df['noisy_error'] == 0).sum() / len(noise_df) * 100
            print(f"  Noise {noise:.3f}: {perfect_rate:5.1f}% have zero baseline")
    
    # H4: Are perfect baselines more common with local strategy?
    if len(perfect) > 0 and 'strategy' in df.columns:
        print("\nH4: Does strategy affect zero baseline rate?")
        for strat in df['strategy'].unique():
            strat_df = df[df['strategy'] == strat]
            perfect_rate = (strat_df['noisy_error'] == 0).sum() / len(strat_df) * 100
            print(f"  {strat:10s}: {perfect_rate:5.1f}% have zero baseline")
    
    if len(perfect) > 0:
        print("\n⚠️  CRITICAL ISSUE: You have experiments with ZERO baseline error!")
        print("\nThis shouldn't happen in a noisy simulation. Possible causes:")
        print("\n1. BUG in baseline calculation:")
        print("   → Check if noisy_error is being calculated correctly")
        print("   → Verify that noise is actually being applied to baseline")
        print("   → Look for division-by-zero protection that might set to 0")
        
        print("\n2. PERFECT CIRCUITS:")
        print("   → Some circuits might be producing correct output despite noise")
        print("   → This could happen for trivial circuits or with lucky noise")
        print("   → But 13.6% seems too high for this explanation")
        
        print("\n3. MEASUREMENT ISSUE:")
        print("   → Check if error calculation uses sufficient shots")
        print("   → With few shots, discrete measurements might occasionally match perfectly")
        print("   → But again, 13.6% seems too high")
        
        print("\n4. CIRCUIT-SPECIFIC BEHAVIOR:")
        print("   → Look at the algorithms with highest zero-baseline rates")
        print("   → Are they particularly simple (e.g., Deutsch-Jozsa with few qubits)?")
        print("   → Some quantum algorithms might be noise-resistant")
        
        print("\n📋 RECOMMENDED ACTIONS:")
        print("\n  A. Check your mqt.py simulation code:")
        print("     - Verify noise is applied in 'no' strategy baseline")
        print("     - Ensure shots parameter is used (not exact simulation)")
        print("     - Look for any error calculation that could return 0")
        
        print("\n  B. Check simulate.py error metric:")
        print("     - How is 'noisy_error' calculated?")
        print("     - Is there a minimum threshold that gets rounded to 0?")
        print("     - Are you using state fidelity or measurement outcomes?")
        
        print("\n  C. Spot-check some zero-baseline experiments:")
        print("     - Manually run a few and verify results")
        print("     - Print intermediate values to see where 0 appears")
        
        print("\n  D. Consider filtering approach:")
        print("     - Remove experiments with noisy_error < 0.001")
        print("     - Report this in methods as excluding 'trivial circuits'")
        print("     - But understand WHY before filtering!")
        
    else:
        print("\n✓ No perfect baseline cases found - this is expected!")

def main():
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = '../results.csv'
    
    print(f"Loading: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} experiments")
    
    investigate_zero_baseline(df)

if __name__ == "__main__":
    main()