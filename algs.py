"""
Algorithm generation module for creating test quantum circuits.
Generates random quantum circuits with various gate types and structures.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, GroverOperator
from typing import Optional, List
import random


def create_random_circuit(num_qubits: int, depth: int, 
                         gate_set: Optional[List[str]] = None,
                         seed: Optional[int] = None) -> QuantumCircuit:
    """
    Create a random quantum circuit with specified number of qubits and depth.
    
    Args:
        num_qubits: Number of qubits in the circuit
        depth: Number of layers of gates
        gate_set: List of gate types to use. If None, uses default set.
        seed: Random seed for reproducibility
    
    Returns:
        Random quantum circuit
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if gate_set is None:
        # Default gate set with common gates
        gate_set = ['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx', 'cz', 's', 't']
    
    qc = QuantumCircuit(num_qubits)
    
    for layer in range(depth):
        # Randomly select qubits and gates for this layer
        for _ in range(num_qubits // 2 + 1):  # Apply multiple gates per layer
            gate = random.choice(gate_set)
            
            if gate in ['h', 'x', 'y', 'z', 's', 't']:
                # Single qubit gates
                qubit = random.randint(0, num_qubits - 1)
                getattr(qc, gate)(qubit)
                
            elif gate in ['rx', 'ry', 'rz']:
                # Parameterized single qubit gates
                qubit = random.randint(0, num_qubits - 1)
                angle = np.random.uniform(0, 2 * np.pi)
                getattr(qc, gate)(angle, qubit)
                
            elif gate in ['cx', 'cz', 'cy']:
                # Two qubit gates
                if num_qubits >= 2:
                    q1 = random.randint(0, num_qubits - 1)
                    q2 = random.randint(0, num_qubits - 1)
                    while q2 == q1:
                        q2 = random.randint(0, num_qubits - 1)
                    getattr(qc, gate)(q1, q2)
    
    return qc


def create_random_clifford_circuit(num_qubits: int, depth: int,
                                   seed: Optional[int] = None) -> QuantumCircuit:
    """
    Create a random Clifford circuit (H, S, CNOT gates only).
    Useful for testing since Clifford circuits can be efficiently simulated.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        seed: Random seed
    
    Returns:
        Random Clifford circuit
    """
    clifford_gates = ['h', 's', 'cx']
    return create_random_circuit(num_qubits, depth, clifford_gates, seed)


def create_random_hardware_efficient_circuit(num_qubits: int, depth: int,
                                            seed: Optional[int] = None) -> QuantumCircuit:
    """
    Create a hardware-efficient ansatz circuit with rotation and entangling layers.
    
    Args:
        num_qubits: Number of qubits
        depth: Number of layers
        seed: Random seed
    
    Returns:
        Hardware-efficient circuit
    """
    if seed is not None:
        np.random.seed(seed)
    
    qc = QuantumCircuit(num_qubits)
    
    for layer in range(depth):
        # Rotation layer
        for q in range(num_qubits):
            qc.ry(np.random.uniform(0, 2*np.pi), q)
            qc.rz(np.random.uniform(0, 2*np.pi), q)
        
        # Entangling layer (linear connectivity)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        
        # Add some randomness with additional CX gates
        if num_qubits > 2:
            for _ in range(num_qubits // 3):
                q1 = random.randint(0, num_qubits - 2)
                q2 = q1 + random.choice([1, 2]) if q1 < num_qubits - 2 else q1 + 1
                if q2 < num_qubits:
                    qc.cx(q1, q2)
    
    return qc


def create_qft_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a Quantum Fourier Transform circuit.
    
    Args:
        num_qubits: Number of qubits
    
    Returns:
        QFT circuit
    """
    qc = QuantumCircuit(num_qubits)
    qft = QFT(num_qubits)
    qc.compose(qft, inplace=True)
    return qc


def create_ghz_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a GHZ (Greenberger-Horne-Zeilinger) state preparation circuit.
    
    Args:
        num_qubits: Number of qubits
    
    Returns:
        GHZ state preparation circuit
    """
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for q in range(num_qubits - 1):
        qc.cx(q, q + 1)
    return qc


def create_w_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Create a W state preparation circuit.
    
    Args:
        num_qubits: Number of qubits
    
    Returns:
        W state preparation circuit
    """
    qc = QuantumCircuit(num_qubits)
    
    # Simple W state preparation (not most efficient but works)
    angles = [np.arccos(np.sqrt(1/k)) for k in range(num_qubits, 0, -1)]
    
    qc.ry(angles[0], 0)
    for i in range(1, num_qubits):
        qc.cry(angles[i], i-1, i)
    
    for i in range(num_qubits - 1):
        qc.cx(i, i+1)
    
    return qc


def create_supremacy_circuit(num_qubits: int, depth: int,
                             seed: Optional[int] = None) -> QuantumCircuit:
    """
    Create a quantum supremacy-style random circuit with all-to-all connectivity.
    
    Args:
        num_qubits: Number of qubits
        depth: Circuit depth
        seed: Random seed
    
    Returns:
        Supremacy-style circuit
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    qc = QuantumCircuit(num_qubits)
    
    single_qubit_gates = [
        lambda q: qc.rx(np.random.uniform(0, 2*np.pi), q),
        lambda q: qc.ry(np.random.uniform(0, 2*np.pi), q),
        lambda q: qc.rz(np.random.uniform(0, 2*np.pi), q),
    ]
    
    for layer in range(depth):
        # Random single qubit gates on all qubits
        for q in range(num_qubits):
            gate = random.choice(single_qubit_gates)
            gate(q)
        
        # Random two-qubit gates
        available_qubits = list(range(num_qubits))
        random.shuffle(available_qubits)
        
        for i in range(0, len(available_qubits) - 1, 2):
            q1, q2 = available_qubits[i], available_qubits[i+1]
            # Use CZ gates (common in supremacy experiments)
            qc.cz(q1, q2)
    
    return qc


def get_algorithm(algorithm_name: str, num_qubits: int, 
                  **kwargs) -> QuantumCircuit:
    """
    Get a specific algorithm circuit by name.
    
    Args:
        algorithm_name: Name of the algorithm
        num_qubits: Number of qubits
        **kwargs: Additional algorithm-specific parameters
    
    Returns:
        Quantum circuit for the specified algorithm
    """
    algorithms = {
        'random': lambda: create_random_circuit(num_qubits, 
                                               kwargs.get('depth', num_qubits * 2),
                                               seed=kwargs.get('seed')),
        'clifford': lambda: create_random_clifford_circuit(num_qubits,
                                                          kwargs.get('depth', num_qubits * 2),
                                                          seed=kwargs.get('seed')),
        'hardware_efficient': lambda: create_random_hardware_efficient_circuit(
            num_qubits, kwargs.get('depth', num_qubits), seed=kwargs.get('seed')),
        'qft': lambda: create_qft_circuit(num_qubits),
        'ghz': lambda: create_ghz_circuit(num_qubits),
        'w_state': lambda: create_w_circuit(num_qubits),
        'supremacy': lambda: create_supremacy_circuit(num_qubits,
                                                     kwargs.get('depth', num_qubits * 2),
                                                     seed=kwargs.get('seed')),
    }
    
    if algorithm_name.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                        f"Available: {list(algorithms.keys())}")
    
    return algorithms[algorithm_name.lower()]()


if __name__ == "__main__":
    # Test the algorithm generation
    print("Testing random circuit generation...")
    
    # Random circuit
    qc_random = create_random_circuit(num_qubits=4, depth=3, seed=42)
    print("\nRandom circuit:")
    print(qc_random)
    
    # QFT circuit
    qc_qft = create_qft_circuit(num_qubits=3)
    print("\nQFT circuit:")
    print(qc_qft)
    
    # GHZ circuit
    qc_ghz = create_ghz_circuit(num_qubits=4)
    print("\nGHZ circuit:")
    print(qc_ghz)
    
    # Hardware-efficient circuit
    qc_he = create_random_hardware_efficient_circuit(num_qubits=4, depth=2, seed=42)
    print("\nHardware-efficient circuit:")
    print(qc_he)