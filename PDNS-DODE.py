"""
PDNS-DODE: A Discrete Hybrid Optimizer for Influence Maximization
Core Framework Pseudocode (Simplified Comments)
Dependencies: networkx >= 2.8, numpy >= 1.21
"""

import networkx as nx
import numpy as np
import random
from typing import List, Set, Tuple


class PDNS_DODE:
    def __init__(self, graph: nx.Graph, k: int, pop_size: int, max_iter: int, 
                 div: float, CR_min: float, CR_max: float):
        """初始化算法参数与网络预计算属性"""
        self.G = graph
        self.k = k
        self.n = pop_size
        self.T_max = max_iter
        self.div = div
        self.CR_min = CR_min
        self.CR_max = CR_max
        
        self.pagerank = nx.pagerank(self.G, alpha=0.85)
        self.PR_max = max(self.pagerank.values())
        self.T = self.PR_max / 5
        self.node_list = list(self.G.nodes())

    def optimize(self) -> List[int]:
        """主优化流程：初始化→迭代优化（全局探索+局部开发）→局部搜索，返回最优种子集"""
        population = self.pagerank_descending_initialization()
        best_solution = None
        best_fitness = -1
        
        for t in range(self.T_max):
            R_T = self.calculate_search_radius(t)
            population = self.discrete_ooa_phase(population, t, R_T)
            population = self.discrete_de_phase(population, t, R_T)
            
            current_best, current_fitness = self.find_best_solution(population)
            if current_fitness > best_fitness:
                best_solution = current_best
                best_fitness = current_fitness
        
        best_solution = self.local_search(best_solution)
        return best_solution

    def pagerank_descending_initialization(self) -> List[List[int]]:
        """生成高质量且多样化的初始种群，提升收敛速度"""
        sorted_nodes = sorted(self.pagerank.items(), key=lambda x: x[1], reverse=True)
        sorted_node_ids = [node for node, _ in sorted_nodes]
        population = []
        
        for i in range(self.n):
            base_seed = sorted_node_ids[:self.k].copy()
            up_bound = min(self.k * (i + 10), len(sorted_node_ids))
            
            for j in range(self.k):
                if random.random() > self.div:
                    replace_idx = random.randint(0, up_bound - 1)
                    base_seed[j] = sorted_node_ids[replace_idx]
            
            base_seed = self._remove_duplicates(base_seed)
            population.append(base_seed)
        
        return population

    def calculate_search_radius(self, current_iter: int) -> float:
        """计算自适应搜索半径，迭代初期大半径探索，后期小半径开发"""
        return 0.2 + ((self.T_max - current_iter) / self.T_max) * 0.8

    def pdns_strategy(self, seed_set: List[int], current_iter: int) -> int:
        """PageRank扩散邻域搜索（PDNS），筛选3跳邻域内高影响力节点作为替换候选"""
        R_T = self.calculate_search_radius(current_iter)
        three_hop_neighbors = self._get_three_hop_neighbors(seed_set)
        filtered_nodes = [n for n in three_hop_neighbors if self.pagerank[n] >= self.T]
        filtered_nodes_sorted = sorted(filtered_nodes, key=lambda x: self.pagerank[x], reverse=True)
        
        candidate_count = max(1, int(len(filtered_nodes_sorted) * R_T))
        candidate_set = filtered_nodes_sorted[:candidate_count]
        
        while True:
            selected_node = random.choice(candidate_set)
            if selected_node not in seed_set:
                return selected_node

    def discrete_ooa_phase(self, population: List[List[int]], current_iter: int, R_T: float) -> List[List[int]]:
        """离散鱼鹰优化（OOA）阶段：全局探索，定位高潜力解区域"""
        new_population = []
        fitness_values = [self.edv_fitness(ind) for ind in population]
        best_fitness = max(fitness_values)
        
        for i, individual in enumerate(population):
            superior_solutions = [
                sol for j, sol in enumerate(population) if fitness_values[j] > fitness_values[i]
            ]
            
            if not superior_solutions:
                new_population.append(individual)
                continue
            
            SF_i = random.choice(superior_solutions)
            I_i = fitness_values[i] / best_fitness if best_fitness != 0 else 0.0
            temp_vector = [1 if node in SF_i else 0 for node in individual]
            
            new_individual = []
            for j, node in enumerate(individual):
                r1 = random.uniform(0.5, 1.0)
                Q_ij = r1 * I_i * temp_vector[j]
                
                if Q_ij >= 0.5:
                    new_individual.append(node)
                else:
                    new_node = self.pdns_strategy(individual, current_iter)
                    new_individual.append(new_node)
            
            new_individual = self._remove_duplicates(new_individual)
            new_population.append(new_individual)
        
        return new_population

    def discrete_de_phase(self, population: List[List[int]], current_iter: int, R_T: float) -> List[List[int]]:
        """离散差分进化（DE）阶段：局部开发，精细优化解"""
        new_population = []
        fitness_values = [self.edv_fitness(ind) for ind in population]
        best_fitness = max(fitness_values)
        
        for i, individual in enumerate(population):
            # 变异操作
            mutated_individual = []
            for node in individual:
                F_ij = self.pagerank[node] / (2 * self.PR_max)
                if random.random() < F_ij:
                    new_node = self.pdns_strategy(individual, current_iter)
                    mutated_individual.append(new_node)
                else:
                    mutated_individual.append(node)
            
            mutated_individual = self._remove_duplicates(mutated_individual)
            
            # 交叉操作
            if best_fitness == 0:
                CR_i = self.CR_min
            else:
                CR_i = self.CR_min + (self.CR_max - self.CR_min) * (fitness_values[i] / best_fitness)
            
            trial_individual = []
            for j in range(self.k):
                current_node = individual[j]
                mutated_node = mutated_individual[j]
                
                if (random.random() <= CR_i) or (self.pagerank[current_node] <= self.pagerank[mutated_node]):
                    trial_individual.append(mutated_node)
                else:
                    trial_individual.append(current_node)
            
            trial_individual = self._remove_duplicates(trial_individual)
            
            # 选择操作
            trial_fitness = self.edv_fitness(trial_individual)
            if trial_fitness > fitness_values[i]:
                new_population.append(trial_individual)
            else:
                new_population.append(individual)
        
        return new_population

    def local_search(self, seed_set: List[int]) -> List[int]:
        """3跳邻域局部搜索，避免局部最优，提升解精度"""
        best_set = seed_set.copy()
        best_fitness = self.edv_fitness(best_set)
        
        for i, node in enumerate(best_set):
            neighbors = self._get_three_hop_neighbors([node])
            candidates = [n for n in neighbors if self.pagerank[n] >= self.T and n not in best_set]
            
            for candidate in candidates:
                new_set = best_set.copy()
                new_set[i] = candidate
                new_fitness = self.edv_fitness(new_set)
                
                if new_fitness > best_fitness:
                    best_set = new_set
                    best_fitness = new_fitness
        
        return best_set

    def _get_three_hop_neighbors(self, seed_set: List[int]) -> Set[int]:
        """工具函数：BFS获取种子集的3跳邻域节点"""
        three_hop = set()
        for seed in seed_set:
            hop1 = set(nx.neighbors(self.G, seed))
            hop2 = set()
            for n1 in hop1:
                hop2.update(nx.neighbors(self.G, n1))
            hop3 = set()
            for n2 in hop2:
                hop3.update(nx.neighbors(self.G, n2))
            three_hop.update(hop1, hop2, hop3)
        three_hop = three_hop - set(seed_set)
        return three_hop

    def edv_fitness(self, seed_set: List[int]) -> float:
        """工具函数：计算适应度（EDV），近似评估影响力传播范围"""
        p = 0.01
        k = len(seed_set)
        
        one_hop_neighbors = set()
        for seed in seed_set:
            one_hop_neighbors.update(nx.neighbors(self.G, seed))
        target_nodes = one_hop_neighbors - set(seed_set)
        
        edv_sum = 0.0
        for node in target_nodes:
            t_i = 0
            for seed in seed_set:
                if self.G.has_edge(seed, node):
                    t_i += 1
            edv_sum += 1 - (1 - p) ** t_i
        
        return k + edv_sum

    def find_best_solution(self, population: List[List[int]]) -> Tuple[List[int], float]:
        """工具函数：筛选种群中的最优解及对应适应度"""
        best_idx = 0
        best_fitness = self.edv_fitness(population[0])
        for i in range(1, len(population)):
            current_fitness = self.edv_fitness(population[i])
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_idx = i
        return population[best_idx], best_fitness

    def _remove_duplicates(self, seed_set: List[int]) -> List[int]:
        """工具函数：移除种子集中的重复节点，保证种子集大小为k"""
        unique_set = list(dict.fromkeys(seed_set))
        while len(unique_set) < self.k:
            candidate = random.choice(self.node_list)
            if candidate not in unique_set:
                unique_set.append(candidate)
        if len(unique_set) > self.k:
            unique_set = unique_set[:self.k]
        return unique_set


# 完整运行示例
if __name__ == '__main__':
    """
    运行指南：
    1. 安装依赖：pip install networkx numpy
    2. 准备数据：边列表格式文件放在工作目录
    3. 配置路径：修改graph_path指向你的网络文件
    4. 调整参数：根据需求修改k、pop_size等
    5. 运行脚本：python PDNS-DODE.py
    """
    graph_path = "social_network_edge_list.txt"
    
    print(f"正在加载网络文件：{graph_path}...")
    G = nx.read_edgelist(graph_path, nodetype=int, create_using=nx.Graph())
    print(f"网络加载完成：节点数={G.number_of_nodes()}, 边数={G.number_of_edges()}")
    
    k = 20
    pop_size = 80
    max_iter = 80
    div = 0.4
    CR_min = 0.2
    CR_max = 0.7
    
    print(f"\n算法参数：k={k}, 种群规模={pop_size}, 最大迭代={max_iter}")
    print("开始优化...")
    pdns_dode = PDNS_DODE(
        graph=G,
        k=k,
        pop_size=pop_size,
        max_iter=max_iter,
        div=div,
        CR_min=CR_min,
        CR_max=CR_max
    )
    optimal_seeds = pdns_dode.optimize()
    optimal_edv = pdns_dode.edv_fitness(optimal_seeds)
    
    print("\n=== 优化结果 ===")
    print(f"最优种子集（节点ID）：{optimal_seeds}")
    print(f"最优种子集的EDV值（影响力近似值）：{round(optimal_edv, 2)}")
    print(f"提示：若需更精确的影响力评估，可基于IC模型运行Monte Carlo模拟（建议10000次）")
