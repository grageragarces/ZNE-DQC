"""
Comprehensive Test Suite for partitioning.py

Tests all major functions to ensure correctness after bug fixes.
Run this after making changes to partitioning.py to verify everything works.
"""

import sys
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit

# Import the fixed partitioning module
sys.path.insert(0, '/home/claude')
from partitioning_original import (
    circuit_to_graph,
    partition_graph,
    find_cut_edges,
    extract_subcircuit,
    reconnect_partitions,
    global_opt,
    local_opt,
    partitioning
)


class Colors:
    """ANSI color codes for pretty output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def create_test_circuit(num_qubits=4, pattern='linear'):
    """Create test circuits with different connectivity patterns."""
    qc = QuantumCircuit(num_qubits)
    
    if pattern == 'linear':
        qc.h(0)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        qc.h(num_qubits - 1)
    elif pattern == 'star':
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
    elif pattern == 'full':
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qc.cx(i, j)
    elif pattern == 'ghz':
        qc.h(0)
        for i in range(1, num_qubits):
            qc.cx(0, i)
    
    return qc


class TestSuite:
    """Comprehensive test suite for partitioning functions."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def test(self, condition, name, error_msg=""):
        """Run a test and report result."""
        if condition:
            print(f"{Colors.GREEN}✓{Colors.END} {name}")
            self.passed += 1
            return True
        else:
            print(f"{Colors.RED}✗{Colors.END} {name}")
            if error_msg:
                print(f"  {Colors.RED}Error: {error_msg}{Colors.END}")
            self.failed += 1
            return False
    
    def warn(self, message):
        """Print a warning."""
        print(f"{Colors.YELLOW}⚠{Colors.END}  {message}")
        self.warnings += 1
    
    def section(self, title):
        """Print section header."""
        print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}")
        print(f"{Colors.BLUE}{Colors.BOLD}{title}{Colors.END}")
        print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    def subsection(self, title):
        """Print subsection header."""
        print(f"\n{Colors.BOLD}{title}{Colors.END}")
        print("-" * 70)
    
    # Test 1: circuit_to_graph
    def test_circuit_to_graph(self):
        self.section("TEST 1: circuit_to_graph")
        
        # Test linear circuit
        qc = create_test_circuit(4, 'linear')
        G, all_gates = circuit_to_graph(qc)
        
        self.test(G.number_of_nodes() == 4, "Linear circuit: correct node count")
        self.test(G.number_of_edges() == 3, "Linear circuit: correct edge count", 
                 f"Got {G.number_of_edges()}")
        self.test(len(all_gates) > 0, "Gates stored in all_gates list")
        
        # Test fully connected
        qc = create_test_circuit(4, 'full')
        G, all_gates = circuit_to_graph(qc)
        
        self.test(G.number_of_edges() == 6, "Fully connected: correct edges (C(4,2)=6)",
                 f"Got {G.number_of_edges()}")
        
        # Test edge attributes
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cz(0, 1)
        G, all_gates = circuit_to_graph(qc)
        
        if G.has_edge(0, 1):
            gates = G[0][1].get('gates', [])
            self.test(len(gates) == 2, "Edge stores multiple gates", 
                     f"Got {len(gates)} gates")
        else:
            self.test(False, "Edge created for two-qubit gate", "No edge found")
    
    # Test 2: partition_graph (THE KEY TEST)
    def test_partition_graph(self):
        self.section("TEST 2: partition_graph (CRITICAL BUG FIX)")
        
        # Create a linear graph
        G = nx.Graph()
        G.add_nodes_from(range(10))
        G.add_edges_from([(i, i+1) for i in range(9)])
        
        self.subsection("Linear Graph (10 nodes)")
        
        test_counts = [2, 3, 4, 5, 6, 8, 10]
        results = []
        
        for num_parts in test_counts:
            partition_map = partition_graph(G, num_parts)
            actual = len(set(partition_map.values()))
            results.append(actual)
            
            # For num_parts < num_nodes, should get exactly num_parts
            if num_parts <= 10:
                self.test(actual == num_parts, 
                         f"Requested {num_parts} → Got {actual}",
                         f"Expected {num_parts}, got {actual}")
        
        unique_results = len(set(results))
        self.test(unique_results >= 5, 
                 f"Creates diverse partition counts ({unique_results} unique)",
                 f"Only {unique_results} unique partition counts - likely still buggy")
        
        # Test fully connected
        self.subsection("Fully Connected Graph (8 nodes)")
        G_full = nx.complete_graph(8)
        
        for num_parts in [2, 4, 6, 8]:
            partition_map = partition_graph(G_full, num_parts)
            actual = len(set(partition_map.values()))
            self.test(actual == num_parts, 
                     f"Fully connected: Requested {num_parts} → Got {actual}")
        
        # Test edge cases
        self.subsection("Edge Cases")
        
        partition_map = partition_graph(G, 1)
        self.test(len(set(partition_map.values())) == 1, 
                 "num_partitions=1 creates 1 partition")
        
        partition_map = partition_graph(G, 15)
        self.test(len(set(partition_map.values())) == 10,
                 "num_partitions > nodes creates node-count partitions")
    
    # Test 3: find_cut_edges
    def test_find_cut_edges(self):
        self.section("TEST 3: find_cut_edges")
        
        # Create simple graph
        G = nx.Graph()
        G.add_nodes_from(range(4))
        G.add_edge(0, 1, gates=[{'name': 'cx', 'qubits': [0, 1]}])
        G.add_edge(1, 2, gates=[{'name': 'cx', 'qubits': [1, 2]}])
        G.add_edge(2, 3, gates=[{'name': 'cx', 'qubits': [2, 3]}])
        
        # Partition: [0,1] | [2,3]
        partition_map = {0: 0, 1: 0, 2: 1, 3: 1}
        cut_edges = find_cut_edges(G, partition_map)
        
        self.test(len(cut_edges) == 1, "Finds correct number of cut edges",
                 f"Expected 1, got {len(cut_edges)}")
        
        if len(cut_edges) > 0:
            q1, q2, gate = cut_edges[0]
            crosses = (partition_map[q1] != partition_map[q2])
            self.test(crosses, "Cut edge crosses partition boundary")
        
        # Test with 3 partitions
        partition_map = {0: 0, 1: 1, 2: 1, 3: 2}
        cut_edges = find_cut_edges(G, partition_map)
        self.test(len(cut_edges) == 2, "3 partitions: correct cut edges",
                 f"Expected 2, got {len(cut_edges)}")
    
    # Test 4: extract_subcircuit
    def test_extract_subcircuit(self):
        self.section("TEST 4: extract_subcircuit")
        
        qc = create_test_circuit(4, 'linear')
        G, all_gates = circuit_to_graph(qc)
        
        # Partition into 2
        partition_map = {0: 0, 1: 0, 2: 1, 3: 1}
        
        sub1 = extract_subcircuit(all_gates, partition_map, 0)
        sub2 = extract_subcircuit(all_gates, partition_map, 1)
        
        self.test(sub1.num_qubits == 2, "Subcircuit 1: correct qubit count",
                 f"Expected 2, got {sub1.num_qubits}")
        self.test(sub2.num_qubits == 2, "Subcircuit 2: correct qubit count",
                 f"Expected 2, got {sub2.num_qubits}")
        self.test(sub1.depth() > 0 or sub2.depth() > 0, "Subcircuits have operations")
    
    # Test 5: reconnect_partitions
    def test_reconnect_partitions(self):
        self.section("TEST 5: reconnect_partitions")
        
        # Create simple subcircuits
        sub1 = QuantumCircuit(2)
        sub1.h(0)
        sub1.cx(0, 1)
        
        sub2 = QuantumCircuit(2)
        sub2.h(0)
        sub2.cx(0, 1)
        
        subcircuits = [sub1, sub2]
        cut_edges = [(1, 2, {'name': 'cx', 'qubits': [1, 2]})]
        partition_map = {0: 0, 1: 0, 2: 1, 3: 1}
        
        # Test both primitives
        for primitive in ['cat', 'tp']:
            final = reconnect_partitions(subcircuits, cut_edges, partition_map, primitive)
            
            self.test(final.num_qubits >= 4, 
                     f"{primitive}: creates circuit with enough qubits",
                     f"Got {final.num_qubits} qubits")
            self.test(final.depth() > 0, 
                     f"{primitive}: circuit has operations")
    
    # Test 6: global_opt
    def test_global_opt(self):
        self.section("TEST 6: global_opt")
        
        qc = create_test_circuit(6, 'linear')
        
        for num_parts in [2, 3, 4]:
            try:
                result = global_opt(qc, num_partitions=num_parts, comm_primitive='cat')
                self.test(result.num_qubits > 0 and result.depth() > 0,
                         f"{num_parts} partitions: creates valid circuit",
                         f"qubits={result.num_qubits}, depth={result.depth()}")
            except Exception as e:
                self.test(False, f"{num_parts} partitions", str(e))
    
    # Test 7: local_opt
    def test_local_opt(self):
        self.section("TEST 7: local_opt")
        
        qc = create_test_circuit(6, 'linear')
        
        for num_parts in [2, 3, 4]:
            try:
                result = local_opt(qc, num_partitions=num_parts, comm_primitive='cat')
                self.test(result.num_qubits > 0 and result.depth() > 0,
                         f"{num_parts} partitions: creates valid circuit",
                         f"qubits={result.num_qubits}, depth={result.depth()}")
            except Exception as e:
                self.test(False, f"{num_parts} partitions", str(e))
    
    # Test 8: partitioning (main function)
    def test_partitioning_main(self):
        self.section("TEST 8: partitioning (main interface)")
        
        qc = create_test_circuit(6, 'linear')
        
        for strategy in ['global', 'local']:
            for num_parts in [2, 4]:
                for primitive in ['cat', 'tp']:
                    try:
                        result = partitioning(
                            qc, 
                            strategy=strategy,
                            num_partitions=num_parts,
                            comm_primitive=primitive
                        )
                        self.test(result.num_qubits > 0,
                                 f"{strategy}/{primitive}/{num_parts}: valid circuit")
                    except Exception as e:
                        self.test(False, f"{strategy}/{primitive}/{num_parts}", str(e))
    
    # Test 9: Edge cases
    def test_edge_cases(self):
        self.section("TEST 9: Edge Cases")
        
        # Single partition
        qc = create_test_circuit(4, 'linear')
        try:
            result = partitioning(qc, num_partitions=1)
            self.test(True, "num_partitions=1 works")
        except Exception as e:
            self.test(False, "num_partitions=1", str(e))
        
        # More partitions than qubits
        try:
            result = partitioning(qc, num_partitions=10)
            self.test(True, "num_partitions > num_qubits works")
        except Exception as e:
            self.test(False, "num_partitions > num_qubits", str(e))
        
        # 2-qubit circuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        try:
            result = partitioning(qc, num_partitions=2)
            self.test(True, "Tiny (2-qubit) circuit works")
        except Exception as e:
            self.test(False, "2-qubit circuit", str(e))
        
        # Circuit with only single-qubit gates
        qc = QuantumCircuit(4)
        for i in range(4):
            qc.h(i)
            qc.rz(np.pi/4, i)
        try:
            result = partitioning(qc, num_partitions=2)
            self.test(True, "Circuit with no 2-qubit gates works")
        except Exception as e:
            self.test(False, "No 2-qubit gates", str(e))
    
    # Test 10: The actual bug scenario
    def test_experiment_scenario(self):
        self.section("TEST 10: Actual Experiment Scenario")
        
        print("Simulating the experiment that was showing num_partitions_tested = 1\n")
        
        # Create a circuit similar to MQT benchmarks
        qc = create_test_circuit(10, 'linear')
        G, all_gates = circuit_to_graph(qc)
        
        partition_counts = [2, 4, 6, 8, 10]
        unique_partition_counts = set()
        
        print(f"{'Requested':<12} {'Actual':<12} {'Status'}")
        print("-" * 70)
        
        for num_parts in partition_counts:
            partition_map = partition_graph(G, num_parts)
            actual = len(set(partition_map.values()))
            unique_partition_counts.add(actual)
            
            status = "✓" if actual == num_parts else "✗"
            print(f"{num_parts:<12} {actual:<12} {status}")
        
        print(f"\nUnique partition counts: {sorted(unique_partition_counts)}")
        print(f"num_partitions_tested would be: {len(unique_partition_counts)}")
        
        self.test(len(unique_partition_counts) >= 4,
                 f"Multiple partition counts tested ({len(unique_partition_counts)})",
                 f"Only {len(unique_partition_counts)} unique - bug may still exist")
        
        self.test(len(unique_partition_counts) == len(partition_counts),
                 "ALL requested partition counts are different",
                 "Some partition counts collapsed to same value")
    
    def summary(self):
        """Print test summary."""
        self.section("TEST SUMMARY")
        
        total = self.passed + self.failed
        success_rate = 100 * self.passed / total if total > 0 else 0
        
        print(f"Passed:   {Colors.GREEN}{self.passed}{Colors.END}")
        print(f"Failed:   {Colors.RED}{self.failed}{Colors.END}")
        if self.warnings > 0:
            print(f"Warnings: {Colors.YELLOW}{self.warnings}{Colors.END}")
        print(f"Total:    {total}")
        print(f"Success:  {success_rate:.1f}%")
        
        if self.failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All tests passed!{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ Some tests failed{Colors.END}")
        
        return self.failed == 0


def main():
    """Run all tests."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}PARTITIONING MODULE - COMPREHENSIVE TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}{'='*70}{Colors.END}\n")
    
    suite = TestSuite()
    
    # Run all tests
    suite.test_circuit_to_graph()
    suite.test_partition_graph()
    suite.test_find_cut_edges()
    suite.test_extract_subcircuit()
    suite.test_reconnect_partitions()
    suite.test_global_opt()
    suite.test_local_opt()
    suite.test_partitioning_main()
    suite.test_edge_cases()
    suite.test_experiment_scenario()
    
    # Summary
    success = suite.summary()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)