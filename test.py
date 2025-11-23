"""
Diagnostic script to verify partitioning is working correctly.
Checks if global vs local strategies produce different circuits.
"""

from algs import create_random_circuit
from partitioning import partitioning
import pandas as pd

def diagnose_partitioning():
    """Check if partitioning actually works differently for global vs local."""
    
    print("="*70)
    print("PARTITIONING DIAGNOSTIC TEST")
    print("="*70)
    
    # Create test circuit
    test_qc = create_random_circuit(num_qubits=6, depth=4, seed=42)
    print(f"\nOriginal circuit:")
    print(f"  Qubits: {test_qc.num_qubits}")
    print(f"  Depth: {test_qc.depth()}")
    print(f"  Gates: {len(test_qc.data)}")
    
    # Test different partition counts
    for num_parts in [2, 4]:
        print(f"\n" + "-"*70)
        print(f"Testing {num_parts} partitions:")
        print("-"*70)
        
        # Global strategy
        print("\nGLOBAL Strategy:")
        try:
            global_qc = partitioning(test_qc, strategy='global', 
                                    num_partitions=num_parts, comm_primitive='cat')
            print(f"  Output qubits: {global_qc.num_qubits}")
            print(f"  Output depth: {global_qc.depth()}")
            print(f"  Output gates: {len(global_qc.data)}")
            
            # Print first few gates
            print(f"  First 5 gates:")
            for i, (inst, qargs, cargs) in enumerate(global_qc.data[:5]):
                print(f"    {i}: {inst.name} on qubits {[q._index if hasattr(q, '_index') else '?' for q in qargs]}")
        except Exception as e:
            print(f"  ERROR: {e}")
            global_qc = None
        
        # Local strategy
        print("\nLOCAL Strategy:")
        try:
            local_qc = partitioning(test_qc, strategy='local', 
                                   num_partitions=num_parts, comm_primitive='cat')
            print(f"  Output qubits: {local_qc.num_qubits}")
            print(f"  Output depth: {local_qc.depth()}")
            print(f"  Output gates: {len(local_qc.data)}")
            
            # Print first few gates
            print(f"  First 5 gates:")
            for i, (inst, qargs, cargs) in enumerate(local_qc.data[:5]):
                print(f"    {i}: {inst.name} on qubits {[q._index if hasattr(q, '_index') else '?' for q in qargs]}")
        except Exception as e:
            print(f"  ERROR: {e}")
            local_qc = None
        
        # Compare
        if global_qc is not None and local_qc is not None:
            print("\nCOMPARISON:")
            if global_qc.depth() == local_qc.depth() and len(global_qc.data) == len(global_qc.data):
                print("  ⚠️  WARNING: Circuits are IDENTICAL!")
                print("  This suggests partitioning isn't working differently.")
            else:
                print("  ✓ Circuits are DIFFERENT (good!)")
                print(f"    Depth difference: {abs(global_qc.depth() - local_qc.depth())}")
                print(f"    Gate count difference: {abs(len(global_qc.data) - len(local_qc.data))}")


def check_noise_simulation():
    """Verify that noise is actually being applied."""
    
    print("\n" + "="*70)
    print("NOISE SIMULATION DIAGNOSTIC TEST")
    print("="*70)
    
    from simulate import create_noise_model, execute_circuit
    from algs import create_random_circuit
    
    # Create simple circuit
    qc = create_random_circuit(4, 2, seed=123)
    
    # Execute without noise
    print("\n1. Executing WITHOUT noise:")
    counts_no_noise = execute_circuit(qc, noise_model=None, shots=1000, seed=42)
    print(f"   Unique outcomes: {len(counts_no_noise)}")
    print(f"   Top 3 results: {sorted(counts_no_noise.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Execute with noise
    print("\n2. Executing WITH noise:")
    noise_model = create_noise_model(local_noise_level=0.01, communication_noise_multiplier=1.0)
    counts_with_noise = execute_circuit(qc, noise_model=noise_model, shots=1000, seed=42)
    print(f"   Unique outcomes: {len(counts_with_noise)}")
    print(f"   Top 3 results: {sorted(counts_with_noise.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Compare
    if counts_no_noise == counts_with_noise:
        print("\n   ⚠️  WARNING: Results are IDENTICAL!")
        print("   Noise might not be applied correctly!")
    else:
        print("\n   ✓ Results are DIFFERENT (noise is working)")


def check_zne_application():
    """Check if ZNE is actually being applied and working."""
    
    print("\n" + "="*70)
    print("ZNE APPLICATION DIAGNOSTIC TEST")
    print("="*70)
    
    from simulate import run_simulation_experiment
    from algs import create_random_circuit
    
    qc = create_random_circuit(6, 3, seed=999)
    
    print("\nRunning experiment with ZNE...")
    result = run_simulation_experiment(
        circuit=qc,
        strategy='global',
        algorithm_name='diagnostic_test',
        local_noise=0.02,
        comm_noise_multiplier=1.1,
        comm_primitive='cat',
        shots=512,
        num_partitions=2,
        apply_zne_flag=True,
        seed=42
    )
    
    print(f"\nResults:")
    print(f"  Noise-free expectation: {result['noise_free_expectation']:.6f}")
    print(f"  Noisy expectation:      {result['noisy_expectation']:.6f}")
    print(f"  ZNE expectation:        {result['zne_expectation']:.6f}")
    print(f"  Noisy error:            {result['noisy_error']:.6f}")
    print(f"  ZNE error:              {result['zne_error']:.6f}")
    print(f"  Error reduction:        {result['error_reduction']:.6f}")
    
    # Check if ZNE did anything
    if result['zne_expectation'] == result['noisy_expectation']:
        print("\n⚠️  WARNING: ZNE expectation = Noisy expectation!")
        print("ZNE might not be applied or is failing!")
    elif abs(result['zne_expectation'] - result['noise_free_expectation']) < abs(result['noisy_expectation'] - result['noise_free_expectation']):
        print("\n✓ ZNE is working (got closer to noise-free)")
    else:
        print("\n⚠️  ZNE made things WORSE (this can happen with wrong extrapolation)")


def analyze_csv_data(csv_file='results.csv'):
    """Analyze the actual CSV data to find patterns."""
    
    print("\n" + "="*70)
    print("CSV DATA ANALYSIS")
    print("="*70)
    
    try:
        df = pd.read_csv(csv_file)
        print(f"\nLoaded {len(df)} experiments from {csv_file}")
        
        # Check if global and local produce identical results
        print("\n1. Checking Global vs Local differences:")
        for parts in sorted(df['num_partitions_tested'].unique()):
            global_data = df[(df['strategy'] == 'global') & (df['num_partitions_tested'] == parts)]
            local_data = df[(df['strategy'] == 'local') & (df['num_partitions_tested'] == parts)]
            
            if len(global_data) > 0 and len(local_data) > 0:
                global_mean = global_data['error_reduction'].mean()
                local_mean = local_data['error_reduction'].mean()
                difference = abs(global_mean - local_mean)
                
                print(f"  {parts} partitions: Global={global_mean:.4f}, Local={local_mean:.4f}, Diff={difference:.4f}")
                
                if difference < 0.001:
                    print(f"    ⚠️  Nearly IDENTICAL! Problem likely.")
        
        # Check partitioned depth
        print("\n2. Checking if partitioning changes circuit depth:")
        depth_changes = df['partitioned_depth'] != df['circuit_depth']
        print(f"  Circuits with depth change: {depth_changes.sum()} / {len(df)}")
        if depth_changes.sum() == 0:
            print("  ⚠️  NO depth changes! Partitioning might not be working!")
        
        # Check error reduction distribution
        print("\n3. Error reduction statistics:")
        print(df['error_reduction'].describe())
        
        if df['error_reduction'].std() < 0.01:
            print("  ⚠️  Very low variance! All results too similar.")
        
    except FileNotFoundError:
        print(f"\n⚠️  File not found: {csv_file}")
    except Exception as e:
        print(f"\n⚠️  Error reading CSV: {e}")


if __name__ == "__main__":
    print("Running comprehensive diagnostics...")
    print("This will help identify what's wrong with your simulation.\n")
    
    # Run all diagnostic tests
    diagnose_partitioning()
    check_noise_simulation()
    check_zne_application()
    
    # Analyze CSV if it exists
    import os
    if os.path.exists('results.csv'):
        analyze_csv_data('results.csv')
    
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    print("\nBased on your plot showing identical global/local lines:")
    print("  Most likely cause: Partitioning returns identical circuits")
    print("  Secondary issue: ZNE may be failing (error reduction ≈ 0)")
    print("\nRun this diagnostic to see exactly what's happening!")