    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_backbone_statistics(G_original, G_backbone, directed=False):
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")
    
    # Preparar layout
    cols = 3 if directed else 2
    fig, axes = plt.subplots(1, cols, figsize=(3.5*cols, 3.5))
    if not isinstance(axes, np.ndarray): axes = [axes] # Handle single plot edge case

    # --- A. Distribución de Pesos P(w) ---
    w_orig = [d["weight"] for u,v,d in G_original.edges(data=True)]
    w_back = [d["weight"] for u,v,d in G_backbone.edges(data=True)]
    
    # Bins logarítmicos
    min_w = min(min(w_orig), min(w_back))
    max_w = max(max(w_orig), max(w_back))
    bins = np.logspace(np.log10(min_w), np.log10(max_w), 25)
    
    ax = axes[0]
    ax.hist(w_orig, bins=bins, color="lightgray", alpha=0.8, density=True, label="Original")
    ax.hist(w_back, bins=bins, histtype="step", color="red", linewidth=1.5, density=True, label="Backbone")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"Weight $\omega$"); ax.set_ylabel(r"$P(\omega)$")
    ax.legend(frameon=False)

    # --- B. Fuerza vs Grado s(k) ---
    # Helper interno para extraer métricas
    def get_sk(G, mode="total"):
        data = []
        for n in G.nodes():
            if mode == "total":
                k, s = G.degree(n), G.degree(n, weight="weight")
            elif mode == "in":
                k, s = G.in_degree(n), G.in_degree(n, weight="weight")
            elif mode == "out":
                k, s = G.out_degree(n), G.out_degree(n, weight="weight")
            if k > 0: data.append({"k": k, "s": s})
        return pd.DataFrame(data)

    if not directed:
        # Undirected: Total s vs k
        df_o = get_sk(G_original)
        df_b = get_sk(G_backbone)
        ax = axes[1]
        ax.scatter(df_o["k"], df_o["s"], s=10, color="lightgray", alpha=0.5)
        ax.scatter(df_b["k"], df_b["s"], s=10, color="red", alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$k$"); ax.set_ylabel(r"$s$")
    else:
        # Directed: in y out separados (o juntos en distintos colores)
        # Aquí mostramos Out-strength vs Out-degree (eje 2) y In-strength vs In-degree (eje 3)
        
        # Plot Out
        df_o_out = get_sk(G_original, "out")
        df_b_out = get_sk(G_backbone, "out")
        ax = axes[1]
        ax.scatter(df_o_out["k"], df_o_out["s"], s=10, color="lightgray", alpha=0.5)
        ax.scatter(df_b_out["k"], df_b_out["s"], s=10, color="red", alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$k_{out}$"); ax.set_ylabel(r"$s_{out}$")

        # Plot In
        df_o_in = get_sk(G_original, "in")
        df_b_in = get_sk(G_backbone, "in")
        ax = axes[2]
        ax.scatter(df_o_in["k"], df_o_in["s"], s=10, color="lightgray", alpha=0.5)
        ax.scatter(df_b_in["k"], df_b_in["s"], s=10, color="blue", alpha=0.6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$k_{in}$"); ax.set_ylabel(r"$s_{in}$")

    sns.despine()
    plt.tight_layout()
    plt.show()
    
    
def plot_alpha_vs_weight(G, alpha_cutoff=0.05):
    """
    Diagnostic plot: Shows the relationship between Weight and Alpha.
    Verifies if strong links actually have low alpha (significance).
    """
    weights = []
    alphas = []
    colors = []
    
    for u, v, d in G.edges(data=True):
        w = d['weight']
        a = d['alpha']
        
        weights.append(w)
        alphas.append(a)
        
        # Color red if it survives (alpha < cutoff), Grey if it dies
        if a < alpha_cutoff:
            colors.append('red') 
        else:
            colors.append('lightgray')

    plt.figure(figsize=(7, 6))
    plt.scatter(weights, alphas, c=colors, alpha=0.5, s=15, edgecolors='none')
    
    plt.xscale('log')
    # Use log scale for Y if alphas are very small, linear otherwise
    plt.yscale('linear') 
    
    plt.axhline(y=alpha_cutoff, color='black', linestyle='--', label=f'Cutoff ({alpha_cutoff})')
    plt.xlabel(r'Weight $\omega$ (Stronger $\to$)')
    plt.ylabel(r'Significance $\alpha$ (Less Random $\to$)')
    plt.title('Which links are being kept?')
    plt.gca().invert_yaxis() # Invert Y so "Best" (Low Alpha) is at the top
    plt.legend(['Cutoff', 'Kept', 'Removed'])
    plt.grid(True, alpha=0.2)
    plt.show()
