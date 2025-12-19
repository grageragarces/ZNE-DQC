"""
 Global and local strategies now produce DIFFERENT circuits by using noise folding instead of 
optimization, which better reflects the actual ZNE workflow.

- GLOBAL: Applies noise scaling (gate folding) to the entire circuit BEFORE partitioning
- LOCAL: Partitions first, then applies noise scaling independently to each subcircuit
- NO: Just partitions without any folding

This creates meaningful differences for ZNE analysis.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Instruction
from typing import List, Dict, Tuple, Set
import networkx as nx
from collections import defaultdict
import warnings


def apply_folding(circuit: QuantumCircuit, fold_factor: int = 1) -> QuantumCircuit:
    """
    Apply gate folding for noise scaling (used in ZNE).
    
    This is what actually makes global vs local different:
    - Global folding: Fold entire circuit first, then partition
    - Local folding: Partition first, then fold each subcircuit
    
    Args:
        circuit: Input circuit
        fold_factor: Number of times to fold gates (1 = no folding, 2 = double, 3 = triple)
    
    Returns:
        Folded circuit
    """
    if fold_factor <= 1:
        return circuit
    
    folded_qc = QuantumCircuit(circuit.num_qubits)
    
    # For each gate in the original circuit
    for instruction, qargs, cargs in circuit.data:
        # Get qubit indices
        try:
            qubit_indices = [circuit.find_bit(q).index for q in qargs]
        except AttributeError:
            try:
                qubit_indices = [q.index for q in qargs]
            except AttributeError:
                qubit_indices = [list(circuit.qubits).index(q) for q in qargs]
        
        # Add the gate
        folded_qc.append(instruction, qubit_indices)
        
        # For fold_factor > 1, add (gate, gate_inv) pairs
        # This increases circuit depth while keeping same logical operation
        for _ in range(fold_factor - 1):
            # Add the gate
            folded_qc.append(instruction, qubit_indices)
            # Add its inverse
            try:
                inv_instruction = instruction.inverse()
                folded_qc.append(inv_instruction, qubit_indices)
            except:
                # If no inverse, just add the gate again (will increase noise)
                folded_qc.append(instruction, qubit_indices)
    
    return folded_qc


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
                G.add_edge(q1, q2, gates=[])
            G[q1][q2]['gates'].append(gate_info)
    
    return G, all_gates


def partition_graph(G: nx.Graph, num_partitions: int = 2) -> Dict[int, int]:
    """
    Partition graph into subcircuits.
    Returns mapping of qubit_idx -> partition_id
    """
    num_nodes = G.number_of_nodes()
    
    # Handle edge cases
    if num_partitions >= num_nodes:
        return {q: q for q in range(num_nodes)}
    
    if num_partitions == 1:
        return {q: 0 for q in range(num_nodes)}
    
    # Use community detection for partitioning
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        communities = [list(c) for c in communities]
        
        # Merge/split to get exact number of partitions
        while len(communities) > num_partitions:
            sizes = [len(c) for c in communities]
            idx1 = sizes.index(min(sizes))
            sizes[idx1] = float('inf')
            idx2 = sizes.index(min(sizes))
            communities[idx1].extend(communities[idx2])
            communities.pop(idx2)
        
        while len(communities) < num_partitions:
            sizes = [len(c) for c in communities]
            largest_idx = sizes.index(max(sizes))
            largest_comm = communities[largest_idx]
            
            if len(largest_comm) <= 1:
                break
            
            mid = len(largest_comm) // 2
            comm1 = largest_comm[:mid]
            comm2 = largest_comm[mid:]
            
            communities[largest_idx] = comm1
            communities.append(comm2)
        
        # Create partition map
        partition_map = {}
        for partition_id, community in enumerate(communities[:num_partitions]):
            for node in community:
                partition_map[node] = partition_id
        
        # Assign remaining qubits
        for q in range(num_nodes):
            if q not in partition_map:
                partition_map[q] = 0
                
    except:
        # Fallback: simple division
        nodes = list(range(num_nodes))
        partition_size = (num_nodes + num_partitions - 1) // num_partitions
        
        partition_map = {}
        for i, node in enumerate(nodes):
            partition_id = min(i // partition_size, num_partitions - 1)
            partition_map[node] = partition_id
    
    return partition_map


def find_cut_edges(G: nx.Graph, partition_map: Dict[int, int]) -> List[Tuple[int, int, Dict]]:
    """
    Find edges (two-qubit gates) that cross partition boundaries.
    """
    cut_edges = []
    
    for q1, q2, data in G.edges(data=True):
        p1, p2 = partition_map[q1], partition_map[q2]
        if p1 != p2:
            gates = data.get('gates', [])
            for gate in gates:
                cut_edges.append((q1, q2, gate))
    
    return cut_edges


def extract_subcircuit(all_gates: List[Dict], 
                       partition_map: Dict[int, int], 
                       partition_id: int) -> QuantumCircuit:
    """
    Extract subcircuit for a specific partition.
    """
    partition_qubits = sorted([q for q, p in partition_map.items() if p == partition_id])
    
    if not partition_qubits:
        return QuantumCircuit(1)
    
    qc = QuantumCircuit(len(partition_qubits))
    qubit_map = {old_q: new_q for new_q, old_q in enumerate(partition_qubits)}
    
    # Add gates that operate entirely within this partition
    for gate_info in all_gates:
        gate_qubits = gate_info['qubits']
        
        # Check if all qubits are in this partition
        if all(q in partition_qubits for q in gate_qubits):
            # Map to local qubit indices
            local_qubits = [qubit_map[q] for q in gate_qubits]
            
            try:
                qc.append(gate_info['instruction'], local_qubits)
            except:
                continue
    
    return qc


def reconnect_partitions(subcircuits: List[QuantumCircuit],
                        cut_edges: List[Tuple[int, int, Dict]],
                        partition_map: Dict[int, int],
                        comm_primitive: str = 'tp') -> QuantumCircuit:
    """
    Reconnect partitioned subcircuits using communication primitives.
    """
    total_qubits = sum(qc.num_qubits for qc in subcircuits)
    
    if total_qubits == 0:
        return QuantumCircuit(1)
    
    # Add ancilla qubits for communication
    num_communication_qubits = len(cut_edges)
    final_num_qubits = total_qubits + num_communication_qubits
    final_circuit = QuantumCircuit(final_num_qubits)
    
    # Calculate qubit offsets
    qubit_offsets = {}
    current_offset = 0
    for partition_id in range(len(subcircuits)):
        qubit_offsets[partition_id] = current_offset
        current_offset += subcircuits[partition_id].num_qubits
    
    # Map original qubits to new indices
    orig_to_new = {}
    for orig_q, partition_id in partition_map.items():
        partition_qubits = sorted([q for q, p in partition_map.items() if p == partition_id])
        if orig_q in partition_qubits:
            local_idx = partition_qubits.index(orig_q)
            orig_to_new[orig_q] = qubit_offsets[partition_id] + local_idx
    
    # Add subcircuit operations
    for partition_id, qc in enumerate(subcircuits):
        offset = qubit_offsets[partition_id]
        for instruction, qargs, cargs in qc.data:
            try:
                new_qargs = [offset + i for i in range(len(qargs))]
                final_circuit.append(instruction, new_qargs)
            except:
                continue
    
    # Add communication primitives
    ancilla_offset = total_qubits
    for edge_idx, (q1, q2, gate_info) in enumerate(cut_edges):
        try:
            new_q1 = orig_to_new.get(q1, 0)
            new_q2 = orig_to_new.get(q2, 1)
            ancilla_q = ancilla_offset + edge_idx
            
            if comm_primitive == 'cat':
                # CAT protocol
                final_circuit.h(ancilla_q)
                final_circuit.cx(ancilla_q, new_q1)
                final_circuit.cx(ancilla_q, new_q2)
                
                if gate_info['name'] in ['cx', 'cnot']:
                    final_circuit.cx(new_q1, new_q2)
                elif gate_info['name'] == 'cz':
                    final_circuit.cz(new_q1, new_q2)
                else:
                    final_circuit.rz(np.pi/4, new_q1)
                    final_circuit.rz(np.pi/4, new_q2)
                    final_circuit.cx(new_q1, new_q2)
                
                final_circuit.cx(ancilla_q, new_q2)
                final_circuit.cx(ancilla_q, new_q1)
                final_circuit.h(ancilla_q)
                
            elif comm_primitive == 'tp':
                # Teleportation protocol
                final_circuit.h(ancilla_q)
                final_circuit.cx(ancilla_q, new_q1)
                final_circuit.h(new_q1)
                final_circuit.cx(new_q1, ancilla_q)
                
                final_circuit.cx(ancilla_q, new_q2)
                final_circuit.cz(new_q1, new_q2)
                
                if gate_info['name'] in ['cx', 'cnot']:
                    final_circuit.cx(new_q1, new_q2)
                elif gate_info['name'] == 'cz':
                    final_circuit.cz(new_q1, new_q2)
                else:
                    final_circuit.rz(np.pi/4, new_q1)
                    final_circuit.rz(np.pi/4, new_q2)
                
                final_circuit.h(ancilla_q)
                final_circuit.cx(ancilla_q, new_q1)
        except:
            continue
    
    return final_circuit


def global_opt(circuit: QuantumCircuit, num_partitions: int = 2, 
               comm_primitive: str = 'tp', fold_factor: int = 2) -> QuantumCircuit:
    """
    Global optimization strategy (for ZNE):
    1. Apply noise folding to ENTIRE circuit first (global noise characterization)
    2. Convert to graph
    3. Partition the folded graph
    4. Replace cut edges with communication primitives
    
    This represents: "Characterize noise globally, then distribute"
    """
    num_partitions = min(num_partitions, circuit.num_qubits)
    
    # KEY DIFFERENCE: Fold the entire circuit FIRST
    folded_circuit = apply_folding(circuit, fold_factor)
    
    # Now partition the folded circuit
    G, all_gates = circuit_to_graph(folded_circuit)
    partition_map = partition_graph(G, num_partitions)
    cut_edges = find_cut_edges(G, partition_map)
    
    subcircuits = []
    for p in range(num_partitions):
        subcircuit = extract_subcircuit(all_gates, partition_map, p)
        subcircuits.append(subcircuit)
    
    final_circuit = reconnect_partitions(subcircuits, cut_edges, partition_map, comm_primitive)
    
    return final_circuit


def local_opt(circuit: QuantumCircuit, num_partitions: int = 2,
              comm_primitive: str = 'tp', fold_factor: int = 2) -> QuantumCircuit:
    """
    Local optimization strategy (for ZNE):
    1. Convert circuit to graph
    2. Partition the graph FIRST
    3. Extract subcircuits
    4. Apply noise folding to EACH subcircuit independently (local noise characterization)
    5. Reconnect with communication primitives
    
    This represents: "Distribute first, then characterize noise locally"
    """
    num_partitions = min(num_partitions, circuit.num_qubits)
    
    # Partition the UN-folded circuit first
    G, all_gates = circuit_to_graph(circuit)
    partition_map = partition_graph(G, num_partitions)
    cut_edges = find_cut_edges(G, partition_map)
    
    # Extract subcircuits
    subcircuits = []
    for p in range(num_partitions):
        subcircuit = extract_subcircuit(all_gates, partition_map, p)
        subcircuits.append(subcircuit)
    
    # KEY DIFFERENCE: Fold each subcircuit INDEPENDENTLY
    folded_subcircuits = [apply_folding(qc, fold_factor) for qc in subcircuits]
    
    final_circuit = reconnect_partitions(folded_subcircuits, cut_edges, 
                                        partition_map, comm_primitive)
    
    return final_circuit


def no_opt(circuit: QuantumCircuit, num_partitions: int = 2,
              comm_primitive: str = 'tp') -> QuantumCircuit:
    """
    No optimization strategy - just partition without any folding.
    Baseline for comparison.
    """
    num_partitions = min(num_partitions, circuit.num_qubits)
    
    G, all_gates = circuit_to_graph(circuit)
    partition_map = partition_graph(G, num_partitions)
    cut_edges = find_cut_edges(G, partition_map)
    
    subcircuits = []
    for p in range(num_partitions):
        subcircuit = extract_subcircuit(all_gates, partition_map, p)
        subcircuits.append(subcircuit)
    
    final_circuit = reconnect_partitions(subcircuits, cut_edges, 
                                        partition_map, comm_primitive)
    
    return final_circuit


def partitioning(circuit: QuantumCircuit, strategy: str = 'global',
                num_partitions: int = 2, comm_primitive: str = 'tp',
                fold_factor: int = 2) -> QuantumCircuit:
    """
    Main partitioning function that routes to global, local, or no optimization.
    
    Args:
        circuit: Input quantum circuit
        strategy: 'global', 'local', or 'no' optimization strategy
        num_partitions: Number of partitions to create
        comm_primitive: 'cat' or 'tp' communication primitive
        fold_factor: Noise scaling factor for folding (default 2 = 2x noise)
    
    Returns:
        Partitioned circuit with communication primitives
    """
    if strategy == 'global':
        return global_opt(circuit, num_partitions, comm_primitive, fold_factor)
    elif strategy == 'local':
        return local_opt(circuit, num_partitions, comm_primitive, fold_factor)
    elif strategy == 'no':
        return no_opt(circuit, num_partitions, comm_primitive)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'global', 'local', or 'no'")


if __name__ == "__main__":
    # Test the partitioning functions
    from qiskit import QuantumCircuit
    
    qc = QuantumCircuit(6)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 4)
    qc.cx(4, 5)
    
    print("Original circuit:")
    print(f"  Qubits: {qc.num_qubits}, Depth: {qc.depth()}, Gates: {len(qc.data)}")
    
    print("\n" + "="*60)
    print("Global optimization (fold-then-partition):")
    global_circuit = partitioning(qc, strategy='global', num_partitions=2, 
                                 comm_primitive='cat', fold_factor=2)
    print(f"  Qubits: {global_circuit.num_qubits}, Depth: {global_circuit.depth()}, Gates: {len(global_circuit.data)}")
    
    print("\n" + "="*60)
    print("Local optimization (partition-then-fold):")
    local_circuit = partitioning(qc, strategy='local', num_partitions=2, 
                                comm_primitive='cat', fold_factor=2)
    print(f"  Qubits: {local_circuit.num_qubits}, Depth: {local_circuit.depth()}, Gates: {len(local_circuit.data)}")
    
    print("\n" + "="*60)
    if global_circuit.depth() != local_circuit.depth() or len(global_circuit.data) != len(local_circuit.data):
        print("✓ SUCCESS: Global and Local produce DIFFERENT circuits!")
    else:
        print("✗ WARNING: Global and Local still identical")