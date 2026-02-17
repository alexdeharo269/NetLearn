import json
import random
import networkx as nx
from networkx.readwrite import json_graph

class Graph:
    def __init__(self, nx_graph=None):
        self.G = nx_graph if nx_graph is not None else nx.Graph()

    @classmethod
    def random(cls, n, k, seed=42):
        G = nx.Graph()
        random.seed(seed)
        
        # 1. Add Nodes
        G.add_nodes_from(range(n))
        
        # 2. Add Edges
        nodes = list(G.nodes())
        for u in nodes:
            possible = [v for v in nodes if v != u]
            if not possible: continue
            
            # Ensure we don't try to sample more than available
            n_edges = random.randint(0, min(k, len(possible)))
            targets = random.sample(possible, n_edges)
            
            for v in targets:
                G.add_edge(u, v, weight=random.random())
        
        # 3. RETURN ONLY AFTER EDGES ARE ADDED
        return cls(G)

    @classmethod
    def scale_free(cls, n=100, m=2, seed=0):
        """Factory: Creates a Barabási-Albert Graph (Scale-Free)."""
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        random.seed(seed)
        for u, v in G.edges():
            G[u][v]['weight'] = random.random()
        return cls(G)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            data = json.load(f)
            # Ensure we load as undirected for this specific filter version
            G = json_graph.node_link_graph(data, directed=False) 
        return cls(G)

    def save_graph(self, graph_path):
        with open(graph_path, "w") as f:
            data = json_graph.node_link_data(self.G, edges="edges", nodes="nodes")
            json.dump(data, f)
            

    def describe_graph(self, min_degree=1, show_centrality=False):
        print(f"\nGraph: {len(self.G.nodes())} nodes, {len(self.G.edges())} edges\n")
        if show_centrality:
            print(self.calc_centrality(self.G, min_degree))

    def calc_centrality(self, graph, min_degree=1):
        sub_graph = graph.copy()
        sub_graph.remove_nodes_from([n for n, d in list(graph.degree) if d < min_degree])
        return nx.betweenness_centrality(sub_graph, weight="weight")