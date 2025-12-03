# PDNS-DODE: A Discrete Osprey Optimization and Differential Evolution Algorithm with PageRank Diffusion Neighborhood Search for Influence Maximization in Social Networks

## Description
This repository provides the official implementation of PDNS-DODE, a discrete hybrid metaheuristic algorithm designed for the Influence Maximization (IM) problem in social networks. The method integrates a discretized Osprey Optimization Algorithm (OOA) with Differential Evolution (DE), enhanced by a novel PageRank-based Diffusion Neighborhood Search (PDNS) strategy. Seven real-world social network datasets are used in this study. The core framework of the algorithm is provided in the `PDNS-DODE.py` file.

## Dataset Information
This work evaluates the algorithm on seven real-world social networks. The **Email-Eu** network contains 1,005 nodes and 25,571 edges, representing internal email communications. The **Blog** network has 3,982 nodes and 6,803 edges, modeling social relationships among blog users. The **Wiki-Vote** network comprises 7,115 nodes and 103,689 edges, capturing Wikipedia administrator elections. The **NetHEPT** network, with 15,229 nodes and 31,376 edges, is an academic co-authorship network from arXiv. Two additional arXiv collaboration networks are used: **CA-AstroPh** (18,772 nodes, 198,110 edges) for Astro Physics and **CA-HepTh** (9,877 nodes, 25,998 edges) for High Energy Physics Theory. Finally, the **Brightkite** location-based social network contains 58,228 nodes and 214,078 edges. All these datasets are publicly available from the Stanford Network Analysis Project (SNAP) and arXiv repositories.

## Code Information
The core algorithm is implemented within the `PDNS_DODE` Python class in `PDNS-DODE.py`. This class encapsulates the complete optimization pipeline, integrating key components such as `pagerank_descending_initialization`, the adaptive `pdns_strategy` for neighborhood search, the `discrete_ooa_phase` for global exploration, the `discrete_de_phase` for local exploitation, and a final `local_search` refinement.

## Methodology
The PDNS-DODE algorithm follows a sequential two-stage optimization pipeline. It begins with a PageRank-descending initialization to ensure population diversity. The global exploration phase employs a discretized Osprey Optimization Algorithm (OOA) to identify promising regions in the solution space. Subsequently, the local exploitation phase utilizes an influence-aware adaptive Differential Evolution (DE) to refine solutions. The core innovation is the PageRank Diffusion Neighborhood Search (PDNS), which dynamically adjusts the candidate search scope within a three-hop neighborhood. A final local search step further optimizes the seed set to escape local optima.

## Environment
The code is developed and tested with Python 3.8+. It requires a few core libraries: `networkx` for graph operations, `numpy` for numerical computations, and optionally `influenceModel` for evaluating the Linear Threshold model as used in the paper's experiments. The experiments reported in the paper were conducted on a machine with an Intel Core i7-8750H CPU and 16 GB RAM.

## Usage Instructions
To run the algorithm:
1. Install the required dependencies, primarily `networkx` and `numpy`, via pip (e.g., `pip install networkx numpy`).
2. Place your graph file (in edge-list format) in the working directory.
3. Modify the `graph` variable in the `__main__` block of `PDNS-DODE.py` to point to your file.
4. Adjust the parameters (`k`, `n`, `max_t`, `div`, `cr_min`, `cr_max`) as needed.
5. Run the script: `python PDNS-DODE.py`.

A complete usage example is provided in the `if __name__ == '__main__':` section of the script.

## Citations
If you use this code, dataset, or any results from the associated research in your work, please cite the following manuscript (currently under review):

```bibtex
@article{zhang2025pdnsdode,
  title={PDNS-DODE: A Discrete Osprey Optimization and Differential Evolution Algorithm with PageRank Diffusion Neighborhood Search for Influence Maximization in Social Networks},
  author={Zhang, Yu and Li, Huan and Mo, Xinyue and Zeng, Xianhong and Na, Xiaoyu},
  year={2025},
  note={Manuscript submitted for publication.}
}
