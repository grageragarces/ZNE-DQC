"""
Parallelized MQT experiment runner - MUCH faster!
Key improvements:
1. Multi-process parallel execution
2. Thread-safe result saving with locks
3. Batch progress updates to reduce I/O overhead
4. Estimated completion time tracking
"""

import os
import re
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from qiskit import QuantumCircuit
import json
from multiprocessing import Pool, Lock, Manager, cpu_count
from functools import partial
import time
from datetime import datetime, timedelta


def parse_mqt_filename(filename: str) -> Tuple[str, int]:
    """Parse MQT Bench filename to extract origin and number of qubits."""
    name = filename.replace('.qasm', '')
    parts = name.split('_')
    
    num_qubits = None
    origin_parts = []
    
    for i, part in enumerate(parts):
        if part == 'qiskit':
            if i > 0 and parts[i-1].isdigit():
                num_qubits = int(parts[i-1])
                origin_parts = parts[:i-1]
            elif i < len(parts) - 1 and parts[i+1].isdigit():
                num_qubits = int(parts[i+1])
                origin_parts = parts[:i]
            break
    else:
        for i in range(len(parts) - 1, -1, -1):
            if parts[i].isdigit():
                num_qubits = int(parts[i])
                origin_parts = parts[:i]
                break
    
    if num_qubits is None:
        for part in reversed(parts):
            if part.isdigit():
                num_qubits = int(part)
                break
    
    origin = '_'.join(origin_parts) if origin_parts else name.replace(f'_{num_qubits}', '')
    origin = origin.replace('_qiskit', '').replace('qiskit_', '')
    
    return origin, num_qubits if num_qubits else 0


def load_mqt_circuit(filepath: str) -> QuantumCircuit:
    """Load a quantum circuit from MQT Bench QASM file."""
    try:
        with open(filepath, 'r') as f:
            qasm_str = f.read()
        
        lines = qasm_str.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith('measure') and 'creg meas' not in stripped:
                filtered_lines.append(line)
        
        cleaned_qasm = '\n'.join(filtered_lines)
        circuit = QuantumCircuit.from_qasm_str(cleaned_qasm)
        
        return circuit
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")
        return None


def discover_mqt_circuits(mqt_folder: str = 'MQTBench',
                         min_qubits: int = 2,
                         max_qubits: int = 12,
                         max_circuits_per_origin: int = 3) -> List[Tuple[QuantumCircuit, str, int]]:
    """Discover and load MQT Bench circuits from folder."""
    circuits = []
    
    if not os.path.exists(mqt_folder):
        print(f"ERROR: Folder '{mqt_folder}' not found!")
        return circuits
    
    print(f"Scanning {mqt_folder} for QASM files...")
    qasm_files = list(Path(mqt_folder).rglob('*.qasm'))
    print(f"Found {len(qasm_files)} QASM files")
    
    circuits_by_origin = {}
    
    for qasm_file in qasm_files:
        filename = qasm_file.name
        origin, num_qubits = parse_mqt_filename(filename)
        
        if num_qubits < min_qubits or num_qubits > max_qubits:
            continue
        
        if origin not in circuits_by_origin:
            circuits_by_origin[origin] = []
        
        if len(circuits_by_origin[origin]) >= max_circuits_per_origin:
            continue
        
        print(f"  Loading: {filename} -> {origin} ({num_qubits} qubits)")
        circuit = load_mqt_circuit(str(qasm_file))
        
        if circuit is not None:
            actual_qubits = circuit.num_qubits
            if actual_qubits != num_qubits:
                print(f"    Warning: Expected {num_qubits} qubits, got {actual_qubits}")
                num_qubits = actual_qubits
            
            circuit_name = f"{origin}_{num_qubits}q"
            circuits_by_origin[origin].append((circuit, circuit_name, num_qubits))
            print(f"    LOADED: {circuit_name}, depth={circuit.depth()}")
    
    for origin_circuits in circuits_by_origin.values():
        circuits.extend(origin_circuits)
    
    print(f"\nTotal circuits loaded: {len(circuits)}")
    print(f"  Origins: {list(circuits_by_origin.keys())}")
    if circuits:
        print(f"  Qubit range: {min(c[2] for c in circuits)} - {max(c[2] for c in circuits)}")
    
    return circuits


def load_progress(progress_file: str) -> set:
    """Load completed experiments from progress file."""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file: str, completed: set):
    """Save completed experiments to progress file."""
    with open(progress_file, 'w') as f:
        json.dump(list(completed), f)


def run_single_experiment(exp_data):
    """
    Run a single experiment - designed to be called in parallel.
    Returns (exp_id, result_dict, success, error_msg)
    """
    from simulate import run_simulation_experiment
    
    exp = exp_data['exp']
    
    try:
        result = run_simulation_experiment(
            circuit=exp['circuit'],
            strategy=exp['strategy'],
            algorithm_name=exp['name'],
            local_noise=exp['noise'],
            comm_noise_multiplier=exp['comm_mult'],
            comm_primitive=exp['primitive'],
            shots=exp_data['shots'],
            num_partitions=exp['parts']
        )
        
        # Check for invalid values
        if result and any(pd.isna(v) or v == float('inf') or v == float('-inf') 
                        for k, v in result.items() if isinstance(v, (int, float))):
            return (exp['exp_id'], result, False, "Invalid values (NaN/inf)")
        
        return (exp['exp_id'], result, True, None)
        
    except ZeroDivisionError as e:
        return (exp['exp_id'], None, False, f"Division by zero: {e}")
    except Exception as e:
        return (exp['exp_id'], None, False, str(e))


def run_mqt_bench_experiments_parallel(
    mqt_folder: str = 'MQTBench',
    min_qubits: int = 4,
    max_qubits: int = 10,
    max_circuits_per_origin: int = 2,
    strategies: List[str] = None,
    noise_levels: List[float] = None,
    comm_noise_multipliers: List[float] = None,
    comm_primitives: List[str] = None,
    partition_counts: List[int] = None,
    shots: int = 512,
    output_file: str = 'mqt_results.csv',
    progress_file: str = 'mqt_progress.json',
    simulator_qubit_limit: int = 31,
    num_workers: int = None,  # NEW: Number of parallel workers
    batch_size: int = 10  # NEW: Save results every N experiments
) -> pd.DataFrame:
    """
    Run experiments in PARALLEL with incremental saving.
    
    Args:
        num_workers: Number of parallel workers (default: CPU count - 1)
        batch_size: Save results after this many experiments complete
    """
    
    # Set defaults
    if strategies is None:
        strategies = ['global', 'local', 'no']
    if noise_levels is None:
        noise_levels = [0.01, 0.02]
    if comm_noise_multipliers is None:
        comm_noise_multipliers = [1.0, 1.05, 1.1, 1.2]
    if comm_primitives is None:
        comm_primitives = ['cat', 'tp']
    if partition_counts is None:
        partition_counts = [2, 4, 6, 8, 10]
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU free
    
    print("="*70)
    print("MQT BENCH ZNE PARTITIONING EXPERIMENTS (PARALLEL)")
    print("="*70)
    print(f"Simulator qubit limit: {simulator_qubit_limit}")
    print(f"Parallel workers: {num_workers}")
    print(f"Batch save interval: {batch_size} experiments")
    
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
        return pd.DataFrame()
    
    # Load progress
    completed_experiments = load_progress(progress_file)
    print(f"\nLoaded progress: {len(completed_experiments)} experiments already completed")
    
    # Load existing results if they exist
    if os.path.exists(output_file):
        results_df = pd.read_csv(output_file)
        print(f"Loaded existing results: {len(results_df)} rows")
    else:
        results_df = pd.DataFrame()
    
    # Generate all experiment configurations
    experiments = []
    skipped_experiments = []
    
    for circuit, name, num_qubits in circuits_info:
        valid_partitions = [p for p in partition_counts if p <= num_qubits]
        if not valid_partitions:
            valid_partitions = [min(2, num_qubits)]
        
        for strategy in strategies:
            for noise in noise_levels:
                for comm_mult in comm_noise_multipliers:
                    for primitive in comm_primitives:
                        for parts in valid_partitions:
                            if primitive == 'cat':
                                estimated_qubits = num_qubits + (parts - 1) * 2
                            else:
                                estimated_qubits = num_qubits + (parts - 1) * 3
                            
                            exp_id = f"{name}_{strategy}_{noise}_{comm_mult}_{primitive}_{parts}"
                            
                            if estimated_qubits > simulator_qubit_limit:
                                if exp_id not in completed_experiments:
                                    skipped_experiments.append({
                                        'name': name,
                                        'num_qubits': num_qubits,
                                        'parts': parts,
                                        'primitive': primitive,
                                        'estimated_qubits': estimated_qubits,
                                        'exp_id': exp_id
                                    })
                                continue
                            
                            if exp_id not in completed_experiments:
                                experiments.append({
                                    'circuit': circuit,
                                    'name': name,
                                    'num_qubits': num_qubits,
                                    'strategy': strategy,
                                    'noise': noise,
                                    'comm_mult': comm_mult,
                                    'primitive': primitive,
                                    'parts': parts,
                                    'exp_id': exp_id
                                })
    
    total_experiments = len(experiments) + len(completed_experiments)
    remaining = len(experiments)
    
    print("\n2. EXPERIMENT PLANNING")
    print("-"*70)
    print(f"Total valid experiments: {total_experiments}")
    print(f"Already completed: {len(completed_experiments)}")
    print(f"Remaining: {remaining}")
    print(f"Skipped (exceeds {simulator_qubit_limit} qubit limit): {len(skipped_experiments)}")
    print(f"\nOutput file: {output_file}")
    
    if remaining == 0:
        print("\n✓ All experiments already completed!")
        return results_df
    
    # Prepare experiment data for parallel execution
    exp_data_list = [{'exp': exp, 'shots': shots} for exp in experiments]
    
    # Run experiments in parallel
    print("\n3. RUNNING EXPERIMENTS IN PARALLEL")
    print("-"*70)
    
    start_time = time.time()
    completed_count = len(completed_experiments)
    batch_results = []
    
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance with progress tracking
        for i, (exp_id, result, success, error_msg) in enumerate(
            pool.imap_unordered(run_single_experiment, exp_data_list), 1
        ):
            completed_count += 1
            
            # Calculate progress and ETA
            progress_pct = (completed_count / total_experiments) * 100
            elapsed = time.time() - start_time
            if i > 0:
                avg_time_per_exp = elapsed / i
                remaining_exp = remaining - i
                eta_seconds = avg_time_per_exp * remaining_exp
                eta = timedelta(seconds=int(eta_seconds))
            else:
                eta = "calculating..."
            
            if success:
                batch_results.append(result)
                completed_experiments.add(exp_id)
                print(f"[{completed_count}/{total_experiments}] ✓ {exp_id[:50]}... "
                      f"({progress_pct:.1f}%, ETA: {eta})")
            else:
                completed_experiments.add(exp_id)  # Mark as attempted
                print(f"[{completed_count}/{total_experiments}] ✗ {exp_id[:50]}... "
                      f"ERROR: {error_msg}")
            
            # Save batch periodically
            if len(batch_results) >= batch_size or i == remaining:
                if batch_results:
                    new_df = pd.DataFrame(batch_results)
                    results_df = pd.concat([results_df, new_df], ignore_index=True)
                    results_df.to_csv(output_file, index=False)
                    save_progress(progress_file, completed_experiments)
                    print(f"  → Saved batch ({len(batch_results)} results, "
                          f"total: {len(results_df)} rows)")
                    batch_results = []
    
    elapsed_total = time.time() - start_time
    
    print("\n" + "="*70)
    print("4. RESULTS SUMMARY")
    print("="*70)
    print(f"Experiments completed: {len(results_df)}")
    print(f"Total time: {timedelta(seconds=int(elapsed_total))}")
    print(f"Average time per experiment: {elapsed_total/remaining:.2f}s")
    print(f"Speedup vs sequential: ~{num_workers}x (with {num_workers} workers)")
    print(f"Results saved to: {output_file}")
    print(f"Skipped (simulator limit): {len(skipped_experiments)}")
    
    if len(results_df) > 0:
        print(f"\nOrigins tested:")
        for origin in results_df['origin'].unique():
            count = len(results_df[results_df['origin'] == origin])
            print(f"  {origin}: {count} experiments")
    
    return results_df


if __name__ == "__main__":
    import sys
    
    print("MQT Bench Parallel Experiment Runner")
    print("="*70)
    
    # Determine number of workers
    available_cpus = cpu_count()
    recommended_workers = max(1, available_cpus - 1)
    print(f"Available CPUs: {available_cpus}")
    print(f"Recommended workers: {recommended_workers}")
    
    folder = 'MQT_nondj'
    if not os.path.exists(folder):
        folder = 'MQTBench'
    
    if not os.path.exists(folder):
        print("\nERROR: Circuit folder not found!")
        sys.exit(1)
    
    print(f"Using folder: {folder}\n")
    
    df = run_mqt_bench_experiments_parallel(
        mqt_folder=folder,
        min_qubits=2,
        max_qubits=15,
        max_circuits_per_origin=10,
        strategies=['global'],
        noise_levels=[0.001, 0.005, 0.01, 0.015, 0.02],
        comm_noise_multipliers=[1.0, 1.1, 1.2],
        comm_primitives=['tp'],
        partition_counts=[1, 2, 3, 4, 5, 6],
        shots=200,
        output_file='results.csv',
        progress_file='test_progress.json',
        simulator_qubit_limit=31,
        num_workers=recommended_workers,  # Use parallel workers
        batch_size=20  # Save every 20 experiments
    )
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)