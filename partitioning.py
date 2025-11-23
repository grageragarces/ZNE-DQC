"""
Partitioning module for quantum circuit optimization with local vs global encoding strategies.
Supports partitioning circuits and replacing cut gates with communication primitives.
Uses graph-based representation for efficiency.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Instruction
from typing import List, Dict, Tuple, Set
import networkx as nx
from collections import defaultdict
import warnings


def circuit_to_graph(circuit: QuantumCircuit) -> Tuple[nx.Graph, List[Dict]]:
    """
    Convert quantum circuit to graph representation.
    Nodes: qubits
    Edges: two-qubit gates
    
    Returns:
        (graph, list of all gate information)
    """
    G = nx.Graph()
    G.add_nodes_from(range(circuit.num_qubits))
    
    all_gates = []  # Store all gates for reconstruction
    
    for instruction, qargs, cargs in circuit.data:
        # Get qubit indices
        try:
            qubit_indices = [circuit.find_bit(q).index for q in qargs]
        except AttributeError:
            try:
                qubit_indices = [q.index for q in qargs]
            except AttributeError:
                qubit_indices = [list(circuit.qubits).index(q) for q in qargs]
        
        gate_info = {
            'name': instruction.name,
            'params': instruction.params,
            'instruction': instruction,
            'qubits': qubit_indices
        }
        all_gates.append(gate_info)
        
        # Add edge for two-qubit gates
        if len(qubit_indices) == 2:
            q1, q2 = qubit_indices
            if not G.has_edge(q1, q2):
                G[q1][q2]['gates'] = []
            G[q1][q2]['gates'].append(gate_info)
    
    return G, all_gates


def optimize_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """
    Optimize quantum circuit using Qiskit's optimization passes.
    This can include gate cancellation, commutation, etc.
    """
    from qiskit.transpiler import PassManager
    
    # Try to import optimization passes - handle version differences
    try:
        # Qiskit 1.x style
        from qiskit.transpiler.passes import (
            Optimize1qGates,
            CommutativeCancellation,
        )
        
        # Create optimization pass manager with available passes
        pm = PassManager([
            CommutativeCancellation(),
            Optimize1qGates(),
        ])
        
    except ImportError:
        # If imports fail, try older style or use basic optimization
        try:
            from qiskit.transpiler.passes import Optimize1qGates
            pm = PassManager([Optimize1qGates()])
        except ImportError:
            # If all else fails, just return the circuit (no optimization)
            import warnings
            warnings.warn("Could not import optimization passes. Skipping circuit optimization.")
            return circuit
    
    # Run optimization
    try:
        return pm.run(circuit)
    except Exception as e:
        # If optimization fails, return original circuit
        import warnings
        warnings.warn(f"Circuit optimization failed: {e}. Using original circuit.")
        return circuit


def partition_graph(G: nx.Graph, num_partitions: int = 2) -> Dict[int, int]:
    """
    Partition graph into subcircuits.
    Returns mapping of qubit_idx -> partition_id
    
    Uses spectral clustering or community detection.
    """
    num_nodes = G.number_of_nodes()
    
    # Handle edge cases
    if num_partitions >= num_nodes:
        # Each qubit in its own partition
        return {q: q for q in range(num_nodes)}
    
    if num_partitions == 1:
        # All qubits in one partition
        return {q: 0 for q in range(num_nodes)}
    
    # Use community detection for partitioning
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # If we got too many communities, merge smallest ones
        while len(communities) > num_partitions:
            # Find two smallest communities and merge
            sizes = [len(c) for c in communities]
            idx1 = sizes.index(min(sizes))
            sizes[idx1] = float('inf')
            idx2 = sizes.index(min(sizes))
            communities[idx1] = communities[idx1].union(communities[idx2])
            communities.pop(idx2)
        
        # Create partition map
        partition_map = {}
        for partition_id, community in enumerate(communities[:num_partitions]):
            for node in community:
                partition_map[node] = partition_id
        
        # Assign any remaining qubits
        for q in range(num_nodes):
            if q not in partition_map:
                partition_map[q] = 0
                
    except:
        # Fallback: simple round-robin
        partition_map = {q: q % num_partitions for q in range(num_nodes)}
    
    return partition_map


def find_cut_edges(G: nx.Graph, partition_map: Dict[int, int]) -> List[Tuple[int, int, Dict]]:
    """
    Find edges (two-qubit gates) that cross partition boundaries.
    Returns list of (q1, q2, gate_info_list).
    """
    cut_edges = []
    
    for q1, q2, data in G.edges(data=True):
        p1, p2 = partition_map[q1], partition_map[q2]
        if p1 != p2:
            # This edge crosses partitions
            gates = data.get('gates', [])
            for gate in gates:
                cut_edges.append((q1, q2, gate))
    
    return cut_edges


def create_cat_state_communication(qc: QuantumCircuit, q1: int, q2: int, 
                                   gate_info: Dict) -> None:
    """
    Replace two-qubit gate with cat state communication primitive (simplified).
    
    Approximates the communication overhead without actual measurements.
    """
    # Cat-entangler: Create entangled state
    qc.h(q1)
    qc.cx(q1, q2)
    
    # Apply operation based on original gate
    if gate_info['name'] in ['cx', 'cnot']:
        qc.cx(q1, q2)
    elif gate_info['name'] == 'cz':
        qc.cz(q1, q2)
    else:
        # Generic gate - approximate with rotations
        qc.rz(np.pi/4, q1)
        qc.rz(np.pi/4, q2)
        qc.cx(q1, q2)
    
    # Cat-disentangler
    qc.cx(q1, q2)
    qc.h(q1)


def create_teleportation_communication(qc: QuantumCircuit, q1: int, q2: int,
                                      gate_info: Dict) -> None:
    """
    Replace two-qubit gate with teleportation communication primitive (simplified).
    
    Approximates the teleportation protocol without actual measurements.
    """
    # Teleportation setup
    qc.h(q1)
    qc.cx(q1, q2)
    
    # Bell measurement approximation (without actual measurement)
    qc.cx(q1, q2)
    qc.h(q1)
    
    # Correction gates based on original gate
    if gate_info['name'] in ['cx', 'cnot']:
        qc.x(q2)
        qc.z(q1)
    elif gate_info['name'] == 'cz':
        qc.z(q2)
        qc.z(q1)
    else:
        # Generic corrections
        qc.rz(np.pi/4, q1)
        qc.rz(np.pi/4, q2)
    
    # Final operations
    qc.h(q2)
    qc.cx(q1, q2)
    

def extract_subcircuit(all_gates: List[Dict], 
                       partition_map: Dict[int, int], 
                       partition_id: int) -> QuantumCircuit:
    """
    Extract subcircuit for a specific partition.
    Only includes gates that operate entirely within the partition.
    """
    # Get qubits in this partition
    partition_qubits = sorted([q for q, p in partition_map.items() if p == partition_id])
    
    if not partition_qubits:
        # Empty partition
        return QuantumCircuit(1)
    
    # Create new circuit with only these qubits
    qc = QuantumCircuit(len(partition_qubits))
    
    # Map old qubit indices to new ones
    qubit_map = {old_q: new_q for new_q, old_q in enumerate(partition_qubits)}
    
    # Add gates that only touch qubits in this partition
    for gate_info in all_gates:
        qubits = gate_info['qubits']
        if all(q in partition_qubits for q in qubits):
            # This gate is entirely in this partition
            new_qubits = [qubit_map[q] for q in qubits]
            try:
                qc.append(gate_info['instruction'], new_qubits)
            except:
                # Skip if gate can't be appended
                continue
    
    return qc


def reconnect_partitions(subcircuits: List[QuantumCircuit], 
                        cut_edges: List[Tuple[int, int, Dict]],
                        partition_map: Dict[int, int],
                        comm_primitive: str = 'cat') -> QuantumCircuit:
    """
    Reconnect partitioned subcircuits using communication primitives.
    
    Args:
        subcircuits: List of optimized subcircuits
        cut_edges: List of cut edges (q1, q2, gate_info)
        partition_map: Mapping of original qubits to partitions
        comm_primitive: 'cat' or 'teleportation'
    """
    # Determine total number of qubits
    total_qubits = sum(qc.num_qubits for qc in subcircuits)
    
    if total_qubits == 0:
        return QuantumCircuit(1)
    
    # Create final circuit
    final_circuit = QuantumCircuit(total_qubits)
    
    # Calculate qubit offset for each partition
    qubit_offsets = {}
    current_offset = 0
    for partition_id in range(len(subcircuits)):
        qubit_offsets[partition_id] = current_offset
        current_offset += subcircuits[partition_id].num_qubits
    
    # Map original qubits to new indices
    orig_to_new = {}
    for orig_q, partition_id in partition_map.items():
        partition_qubits = sorted([q for q, p in partition_map.items() if p == partition_id])
        local_idx = partition_qubits.index(orig_q)
        orig_to_new[orig_q] = qubit_offsets[partition_id] + local_idx
    
    # Add all subcircuit operations
    for partition_id, qc in enumerate(subcircuits):
        offset = qubit_offsets[partition_id]
        for instruction, qargs, cargs in qc.data:
            # Get local qubit indices
            try:
                local_indices = [offset + i for i in range(len(qargs))]
                final_circuit.append(instruction, local_indices)
            except:
                continue
    
    # Add communication primitives for cut edges
    for q1, q2, gate_info in cut_edges:
        try:
            new_q1 = orig_to_new[q1]
            new_q2 = orig_to_new[q2]
            
            if comm_primitive == 'cat':
                create_cat_state_communication(final_circuit, new_q1, new_q2, gate_info)
            elif comm_primitive == 'teleportation':
                create_teleportation_communication(final_circuit, new_q1, new_q2, gate_info)
        except:
            # Skip problematic communication primitive
            continue
    
    return final_circuit
    
    return final_circuit


def global_opt(circuit: QuantumCircuit, num_partitions: int = 2, 
               comm_primitive: str = 'cat') -> QuantumCircuit:
    """
    Global optimization strategy:
    1. Optimize entire circuit first
    2. Convert to graph
    3. Partition the graph
    4. Replace cut edges with communication primitives
    
    Args:
        circuit: Input quantum circuit
        num_partitions: Number of partitions to create
        comm_primitive: 'cat' or 'teleportation'
    
    Returns:
        Partitioned and reconnected circuit
    """
    # Ensure valid number of partitions
    num_partitions = min(num_partitions, circuit.num_qubits)
    num_partitions = max(1, num_partitions)
    
    # Step 1: Global optimization
    optimized_circuit = optimize_circuit(circuit)
    
    # Step 2: Convert to graph
    G, all_gates = circuit_to_graph(optimized_circuit)
    
    # Step 3: Partition
    partition_map = partition_graph(G, num_partitions)
    
    # Step 4: Find cut edges
    cut_edges = find_cut_edges(G, partition_map)
    
    # Step 5: Extract subcircuits
    subcircuits = []
    for p in range(num_partitions):
        subcircuit = extract_subcircuit(all_gates, partition_map, p)
        subcircuits.append(subcircuit)
    
    # Step 6: Reconnect with communication primitives
    final_circuit = reconnect_partitions(subcircuits, cut_edges, partition_map, comm_primitive)
    
    return final_circuit


def local_opt(circuit: QuantumCircuit, num_partitions: int = 2,
              comm_primitive: str = 'cat') -> QuantumCircuit:
    """
    Local optimization strategy:
    1. Convert circuit to graph
    2. Partition the graph
    3. Extract subcircuits
    4. Optimize each subcircuit independently
    5. Reconnect with communication primitives
    
    Args:
        circuit: Input quantum circuit
        num_partitions: Number of partitions to create
        comm_primitive: 'cat' or 'teleportation'
    
    Returns:
        Partitioned and reconnected circuit
    """
    # Ensure valid number of partitions
    num_partitions = min(num_partitions, circuit.num_qubits)
    num_partitions = max(1, num_partitions)
    
    # Step 1: Convert to graph
    G, all_gates = circuit_to_graph(circuit)
    
    # Step 2: Partition
    partition_map = partition_graph(G, num_partitions)
    
    # Step 3: Find cut edges
    cut_edges = find_cut_edges(G, partition_map)
    
    # Step 4: Extract subcircuits
    subcircuits = []
    for p in range(num_partitions):
        subcircuit = extract_subcircuit(all_gates, partition_map, p)
        subcircuits.append(subcircuit)
    
    # Step 5: Optimize each subcircuit independently
    optimized_subcircuits = [optimize_circuit(qc) for qc in subcircuits]
    
    # Step 6: Reconnect with communication primitives
    final_circuit = reconnect_partitions(optimized_subcircuits, cut_edges, 
                                        partition_map, comm_primitive)
    
    return final_circuit


def partitioning(circuit: QuantumCircuit, strategy: str = 'global',
                num_partitions: int = 2, comm_primitive: str = 'cat') -> QuantumCircuit:
    """
    Main partitioning function that routes to global or local optimization.
    
    Args:
        circuit: Input quantum circuit
        strategy: 'global' or 'local' optimization strategy
        num_partitions: Number of partitions to create
        comm_primitive: 'cat' or 'teleportation' communication primitive
    
    Returns:
        Partitioned circuit with communication primitives
    """
    if strategy == 'global':
        return global_opt(circuit, num_partitions, comm_primitive)
    elif strategy == 'local':
        return local_opt(circuit, num_partitions, comm_primitive)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'global' or 'local'")


if __name__ == "__main__":
    # Test the partitioning functions
    from qiskit import QuantumCircuit
    
    # Create a simple test circuit
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.h(3)
    
    print("Original circuit:")
    print(qc)
    
    print("\n" + "="*50)
    print("Global optimization:")
    global_circuit = partitioning(qc, strategy='global', comm_primitive='cat')
    print(global_circuit)
    
    print("\n" + "="*50)
    print("Local optimization:")
    local_circuit = partitioning(qc, strategy='local', comm_primitive='cat')
    print(local_circuit)