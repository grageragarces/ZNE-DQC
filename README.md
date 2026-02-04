# Distributed Quantum Error Mitigation: Global and Local ZNE Encodings

This repository contains the code and data used in the upcoming IEEE INFOCOM 2025 QUNAP publication:

**Distributed Quantum Error Mitigation: Global and Local ZNE Encodings**  
Maria Gragera Garces  
Quantum Software Lab, University of Edinburgh  

## Paper Abstract

Errors are the primary bottleneck preventing practical quantum computing.  
This challenge is exacerbated in the distributed quantum computing regime, where quantum networks introduce additional communication-induced noise.  
While error mitigation techniques such as Zero Noise Extrapolation (ZNE) have proven effective for standalone quantum processors, their behavior in distributed architectures is not yet well understood.  

We investigate ZNE in this setting by comparing:

- **Global optimization**: ZNE is applied prior to circuit partitioning.  
- **Local optimization**: ZNE is applied independently to each sub-circuit.  

Partitioning is performed on a monolithic circuit, which is then transformed into a distributed implementation by inserting noisy teleportation-based communication primitives between sub-circuits.  
We evaluate both approaches across varying numbers of quantum processing units (QPUs) and under heterogeneous local and network noise conditions.  

Our results demonstrate that Global ZNE exhibits superior scalability, achieving error reductions of up to **48%** across six QPUs.  
Moreover, we observe counterintuitive noise behavior, where increasing the number of QPUs improves mitigation effectiveness despite higher communication overhead.  

These findings highlight fundamental trade-offs in distributed quantum error mitigation and raise new questions regarding the interplay between circuit structure, partitioning strategies, and network noise.  


