"""
Script to run ZNE partitioning experiments on MQT Bench circuits.
Loads circuits from QASM files in the MQTBench folder.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from qiskit import QuantumCircuit

from simulate import run_batch_experiments


def parse_mqt_filename(filename: str) -> Tuple[str, int]:
    """
    Parse MQT Bench filename to extract origin and number of qubits.
    
    Example: "ae_indep_qiskit_2.qasm" -> ("ae_indep", 2)
    
    Args:
        filename: QASM filename
        
    Returns:
        (origin_name, num_qubits)
    """
    # Remove .qasm extension
    name = filename.replace('.qasm', '')
    
    # Split by underscore and find the last number
    parts = name.split('_')
    
    # The last part before or after 'qiskit' should be the number of qubits
    num_qubits = None
    origin_parts = []
    
    for i, part in enumerate(parts):
        if part == 'qiskit':
            # Number might be before or after 'qiskit'
            if i > 0 and parts[i-1].isdigit():
                num_qubits = int(parts[i-1])
                origin_parts = parts[:i-1]
            elif i < len(parts) - 1 and parts[i+1].isdigit():
                num_qubits = int(parts[i+1])
                origin_parts = parts[:i]
            break
    else:
        # No 'qiskit' found, try to find last number
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                num_qubits = int(parts[i])
                origin_parts = parts[:i]
                break
    
    if num_qubits is None:
        # Fallback: try to find any number
        for part in reversed(parts):
            if part.isdigit():
                num_qubits = int(part)
                break
    
    # Construct origin name
    origin = '_'.join(origin_parts) if origin_parts else name.replace(f'_{num_qubits}', '')
    
    # Remove 'qiskit' from origin if present
    origin = origin.replace('_qiskit', '').replace('qiskit_', '')
    
    return origin, num_qubits if num_qubits else 0


def load_mqt_circuit(filepath: str) -> QuantumCircuit:
    """
    Load a quantum circuit from MQT Bench QASM file.
    
    Args:
        filepath: Path to .qasm file
        
    Returns:
        QuantumCircuit
    """
    try:
        # Remove measurements from the QASM file before loading
        # This prevents issues with mid-circuit measurements
        with open(filepath, 'r') as f:
            qasm_str = f.read()
        
        # Remove measurement lines (we'll add them back in execute_circuit)
        lines = qasm_str.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep all lines except measure and meas declarations
            if not stripped.startswith('measure') and 'creg meas' not in stripped:
                filtered_lines.append(line)
        
        cleaned_qasm = '\n'.join(filtered_lines)
        
        # Load circuit from cleaned QASM string
        circuit = QuantumCircuit.from_qasm_str(cleaned_qasm)
        
        return circuit
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def discover_mqt_circuits(mqt_folder: str = 'MQTBench',
                         min_qubits: int = 2,
                         max_qubits: int = 12,
                         max_circuits_per_origin: int = 3) -> List[Tuple[QuantumCircuit, str, int]]:
    """
    Discover and load MQT Bench circuits from folder.
    
    Args:
        mqt_folder: Path to MQTBench folder
        min_qubits: Minimum number of qubits
        max_qubits: Maximum number of qubits
        max_circuits_per_origin: Maximum circuits to load per algorithm type
        
    Returns:
        List of (circuit, name, num_qubits) tuples
    """
    circuits = []
    
    if not os.path.exists(mqt_folder):
        print(f"ERROR: Folder '{mqt_folder}' not found!")
        print(f"Please ensure MQTBench folder exists in current directory.")
        return circuits
    
    print(f"Scanning {mqt_folder} for QASM files...")
    
    # Find all .qasm files
    qasm_files = list(Path(mqt_folder).rglob('*.qasm'))
    print(f"Found {len(qasm_files)} QASM files")
    
    # Group by origin to limit circuits per type
    circuits_by_origin = {}
    
    for qasm_file in qasm_files:
        filename = qasm_file.name
        origin, num_qubits = parse_mqt_filename(filename)
        
        # Filter by qubit count
        if num_qubits < min_qubits or num_qubits > max_qubits:
            continue
        
        # Limit circuits per origin
        if origin not in circuits_by_origin:
            circuits_by_origin[origin] = []
        
        if len(circuits_by_origin[origin]) >= max_circuits_per_origin:
            continue
        
        # Load circuit
        print(f"  Loading: {filename} -> {origin} ({num_qubits} qubits)")
        circuit = load_mqt_circuit(str(qasm_file))
        
        if circuit is not None:
            # Verify qubit count matches
            actual_qubits = circuit.num_qubits
            if actual_qubits != num_qubits:
                print(f"    Warning: Expected {num_qubits} qubits, got {actual_qubits}")
                num_qubits = actual_qubits
            
            circuit_name = f"{origin}_{num_qubits}q"
            circuits_by_origin[origin].append((circuit, circuit_name, num_qubits))
            print(f"    ✓ Loaded: {circuit_name}, depth={circuit.depth()}")
    
    # Flatten the dictionary
    for origin_circuits in circuits_by_origin.values():
        circuits.extend(origin_circuits)
    
    print(f"\n✓ Total circuits loaded: {len(circuits)}")
    print(f"  Origins: {list(circuits_by_origin.keys())}")
    print(f"  Qubit range: {min(c[2] for c in circuits)} - {max(c[2] for c in circuits)}")
    
    return circuits


def run_mqt_bench_experiments(mqt_folder: str = 'MQTBench',
                              min_qubits: int = 4,
                              max_qubits: int = 10,
                              max_circuits_per_origin: int = 2,
                              strategies: List[str] = None,
                              noise_levels: List[float] = None,
                              comm_noise_multipliers: List[float] = None,
                              comm_primitives: List[str] = None,
                              partition_counts: List[int] = None,
                              shots: int = 512,
                              output_file: str = 'mqt_results.csv') -> pd.DataFrame:
    """
    Run experiments on MQT Bench circuits.
    
    Args:
        mqt_folder: Path to MQTBench folder containing .qasm files
        min_qubits: Minimum circuit size to test
        max_qubits: Maximum circuit size to test
        max_circuits_per_origin: Max circuits per algorithm type
        strategies: List of strategies ('global', 'local')
        noise_levels: Noise levels to test
        comm_noise_multipliers: Communication noise multipliers
        comm_primitives: Communication primitives to test
        partition_counts: Partition counts to test
        shots: Number of shots per experiment
        output_file: Output CSV file
        
    Returns:
        DataFrame with results
    """
    print("="*70)
    print("MQT BENCH ZNE PARTITIONING EXPERIMENTS")
    print("="*70)
    
    # Discover circuits
    print("\n1. DISCOVERING CIRCUITS")
    print("-"*70)
    circuits_info = discover_mqt_circuits(
        mqt_folder=mqt_folder,
        min_qubits=min_qubits,
        max_qubits=max_qubits,
        max_circuits_per_origin=max_circuits_per_origin
    )
    
    if not circuits_info:
        print("\nERROR: No circuits found!")
        print("Please check:")
        print(f"  1. MQTBench folder exists at: {os.path.abspath(mqt_folder)}")
        print(f"  2. Folder contains .qasm files")
        print(f"  3. Circuits are in qubit range {min_qubits}-{max_qubits}")
        return pd.DataFrame()
    
    # Convert to format expected by run_batch_experiments
    circuits = [(circuit, name) for circuit, name, _ in circuits_info]
    
    # Run experiments
    print("\n2. RUNNING EXPERIMENTS")
    print("-"*70)
    print(f"Configuration:")
    print(f"  Circuits: {len(circuits)}")
    print(f"  Strategies: {strategies or ['global', 'local']}")
    print(f"  Partition counts: {partition_counts or [2, 4, 6, 8, 10]}")
    print(f"  Noise levels: {noise_levels or [0.01]}")
    print(f"  Communication multipliers: {comm_noise_multipliers or [1.0, 1.1]}")
    print(f"  Primitives: {comm_primitives or ['cat', 'teleportation']}")
    print(f"  Shots: {shots}")
    print("")
    
    df = run_batch_experiments(
        circuits=circuits,
        strategies=strategies,
        noise_levels=noise_levels,
        comm_noise_multipliers=comm_noise_multipliers,
        comm_primitives=comm_primitives,
        partition_counts=partition_counts,
        shots=shots,
        output_file=output_file
    )
    
    print("\n3. RESULTS SUMMARY")
    print("-"*70)
    print(f"✓ Experiments completed: {len(df)}")
    print(f"✓ Results saved to: {output_file}")
    
    if len(df) > 0:
        print(f"\nOrigins tested:")
        for origin in df['origin'].unique():
            count = len(df[df['origin'] == origin])
            print(f"  {origin}: {count} experiments")
        
        print(f"\nPartition counts tested:")
        for parts in sorted(df['num_partitions_tested'].unique()):
            count = len(df[df['num_partitions_tested'] == parts])
            print(f"  {int(parts)} partitions: {count} experiments")
    
    return df


def quick_mqt_test(mqt_folder: str = 'MQTBench', 
                   output_file: str = 'mqt_quick_test.csv') -> pd.DataFrame:
    """
    Quick test with minimal configuration to verify MQT Bench circuits work.
    
    Args:
        mqt_folder: Path to MQTBench folder
        output_file: Output file
        
    Returns:
        DataFrame with results
    """
    print("Running quick MQT Bench test...")
    
    return run_mqt_bench_experiments(
        mqt_folder=mqt_folder,
        min_qubits=4,
        max_qubits=8,
        max_circuits_per_origin=1,  # Just 1 circuit per type
        strategies=['global'],  # Just one strategy
        noise_levels=[0.01],  # Just one noise level
        comm_noise_multipliers=[1.0],  # Just one multiplier
        comm_primitives=['cat'],  # Just one primitive
        partition_counts=[2, 4],  # Just two partition counts
        shots=256,  # Fewer shots for speed
        output_file=output_file
    )


if __name__ == "__main__":
    import sys
    
    print("MQT Bench Circuit Experiment Runner")
    print("="*70)
    
    # Check if MQTBench folder exists
    if not os.path.exists('MQTBench'):
        print("\n⚠️  MQTBench folder not found in current directory!")
        print("\nPlease:")
        print("  1. Download MQT Bench circuits from:")
        print("     https://www.cda.cit.tum.de/mqtbench/")
        print("  2. Place .qasm files in a folder named 'MQTBench'")
        print("  3. Run this script again")
        print("\nExample structure:")
        print("  ./MQTBench/")
        print("    ├── ae_indep_qiskit_2.qasm")
        print("    ├── qft_qiskit_4.qasm")
        print("    └── ...")
        sys.exit(1)
    
    print("\nOptions:")
    print("  1. Quick test (1-2 circuits, 2 partition counts) - ~5 minutes")
    print("  2. Medium test (2-3 circuits per origin, 4 partition counts) - ~30 minutes")
    print("  3. Full experiment (3 circuits per origin, all partitions) - ~1-2 hours")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\nRunning quick test...")
        df = quick_mqt_test('MQTBench', 'mqt_quick_test.csv')
        
    elif choice == '2':
        print("\nRunning medium test...")
        df = run_mqt_bench_experiments(
            mqt_folder='MQTBench',
            min_qubits=4,
            max_qubits=10,
            max_circuits_per_origin=2,
            strategies=['global', 'local'],
            noise_levels=[0.01],
            comm_noise_multipliers=[1.0, 1.1],
            comm_primitives=['cat', 'teleportation'],
            partition_counts=[2, 4, 6, 8],
            shots=512,
            output_file='mqt_medium_results.csv'
        )
        
    elif choice == '3':
        print("\nRunning full experiment...")
        confirm = input("This will take 1-2 hours. Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            df = run_mqt_bench_experiments(
                mqt_folder='MQTBench',
                min_qubits=4,
                max_qubits=130,
                max_circuits_per_origin=3,
                strategies=['global', 'local'],
                noise_levels=[0.01, 0.02],
                comm_noise_multipliers=[1.0, 1.05, 1.1, 1.2],
                comm_primitives=['cat', 'teleportation'],
                partition_counts=[2, 4, 6, 8, 10],
                shots=1024,
                output_file='mqt_full_results.csv'
            )
        else:
            print("Cancelled.")
            sys.exit(0)
    else:
        print("Invalid choice.")
        sys.exit(1)
    
    # Show summary
    if len(df) > 0:
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
        print(f"✓ Total experiments: {len(df)}")
        print(f"✓ Unique circuits: {df['origin'].nunique()}")
        print(f"✓ Partition counts: {sorted(df['num_partitions_tested'].unique())}")
        print("\nNext steps:")
        print("  1. Run: python plot_error_reduction.py")
        print("  2. Analyze the results in the CSV file")
    else:
        print("\n⚠️  No experiments completed!")
        print("Check for errors above.")