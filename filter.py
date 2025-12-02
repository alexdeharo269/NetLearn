import networkx as nx

def calculate_alpha(weight, strength, degree):
    """Analytical solution for the disparity filter."""
    if degree <= 1: return 1.0
    p_ij = weight / strength
    # Numerical stability check
    if p_ij >= 1.0: p_ij = 0.999999
    return (1.0 - p_ij)**(degree - 1)

def apply_disparity_filter(G):
    """Modifies G in-place to add 'alpha' attributes."""
    strength = dict(G.degree(weight='weight'))
    degrees = dict(G.degree())
    
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 0.0)
        alpha_u = calculate_alpha(w, strength.get(u,0), degrees.get(u,1))
        alpha_v = calculate_alpha(w, strength.get(v,0), degrees.get(v,1))
        data['alpha'] = min(alpha_u, alpha_v)

def get_backbone(G, alpha_t):
    """Returns a NEW networkx graph with edges where alpha < alpha_t."""
    backbone = nx.Graph()
    backbone.add_nodes_from(G.nodes(data=True))
    
    edges_to_keep = [
        (u, v, data) for u, v, data in G.edges(data=True) 
        if data.get('alpha', 1.0) < alpha_t
    ]
    backbone.add_edges_from(edges_to_keep)
    
    # Optional: remove isolated nodes for cleaner visualization
    backbone.remove_nodes_from(list(nx.isolates(backbone)))
    return backbone