import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

class HyperbolicMapping:
    """
    Implementación del modelo S1/H2 para redes complejas.
    Referencia: Allard & Serrano (2020).
    """
    def __init__(self, G):
        self.G = G
        self.nodes = list(G.nodes())
        self.k = np.array([G.degree(n) for n in self.nodes])
        self.N = len(self.nodes)
        
    def get_radial_coords(self, beta=1.5):
        """
        Calcula las coordenadas radiales r_i.
        Los hubs (mayor grado) se sitúan cerca del centro (r bajo)[cite: 131, 432].
        """
        R = 2 * np.log(self.N) # Radio del disco aproximado [cite: 118]
        # r_i = R - 2*ln(k_i) asegura la jerarquía de hubs [cite: 128]
        r = R - 2 * np.log(self.k / np.min(self.k) + 1e-6)
        return np.maximum(r, 0)

    @staticmethod
    def hyperbolic_distance(r1, theta1, r2, theta2):
        """
        Distancia geodésica en el disco hiperbólico (Curvatura K = -1)[cite: 118].
        """
        dtheta = np.pi - np.abs(np.pi - np.abs(theta1 - theta2))
        ch = np.cosh(r1) * np.cosh(r2) - np.sinh(r1) * np.sinh(r2) * np.cos(dtheta)
        
        return np.arccosh(np.maximum(ch, 1.0))

    def plot_disk(self, r, theta, color_data=None, cmap='viridis'):
        """
        Visualización estilo 'paper' en disco hiperbólico[cite: 394, 405].
        """
        sns.set_context("paper")
        sns.set_style("white")
        
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='polar')
        
        # Plot de nodos
        scatter = ax.scatter(theta, r, c=color_data, cmap=cmap, 
                             s=40, alpha=0.8, edgecolors='none')
        
        # Estética compacta solicitada
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim(0, np.max(r) * 1.05)
        ax.grid(True, linestyle=':', alpha=0.4)
        
        if color_data is not None:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Age Correlation / Disparity', fontsize=10)
            
        plt.tight_layout()
        return fig
    
    

def compute_greedy_routing_metrics(adj, r, theta):
    """Calcula Success Rate y Stretch para un sujeto[cite: 88, 92]."""
    N = adj.shape[0]
    G = nx.from_numpy_array(adj)
    
    # Precalculamos distancias hiperbólicas exactas [cite: 118]
    # Usamos una matriz para optimizar la búsqueda de vecinos
    success_count = 0
    total_stretch = []
    
    # Muestreamos pares para eficiencia si N es grande, o usamos todos en AAL90
    for s in range(N):
        for t in range(N):
            if s == t: continue
            
            curr = s
            path = [curr]
            visited = {curr}
            
            # Protocolo GR: ir al vecino más cercano al target 't' 
            while curr != t:
                neighbors = list(G.neighbors(curr))
                if not neighbors: break
                
                # Distancia hiperbólica de vecinos a 't' [cite: 118]
                dists = [HyperbolicMapping.hyperbolic_distance(r[n], theta[n], r[t], theta[t]) 
                         for n in neighbors]
                best_neighbor = neighbors[np.argmin(dists)]
                
                if best_neighbor in visited: break # Bucle (Fallo) [cite: 86]
                
                curr = best_neighbor
                path.append(curr)
                visited.add(curr)
                
            if curr == t:
                success_count += 1
                # Stretch: Longitud camino / camino más corto 
                short_path = nx.shortest_path_length(G, s, t)
                total_stretch.append(len(path)-1 / short_path if short_path > 0 else 1)
                
    sr = success_count / (N * (N - 1)) # Success Rate [cite: 91]
    avg_stretch = np.mean(total_stretch) if total_stretch else np.nan
    return sr, avg_stretch
