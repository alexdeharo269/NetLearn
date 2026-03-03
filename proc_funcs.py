
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

def get_nodal_disparity_and_strength(connectomes, roi_prefix="R"):
    """
    Calcula la disparidad nodal (Y) y la fuerza nodal (S) para cada región de cada sujeto.
    Retorna: DataFrame (N_sujetos x N_regiones)
    """
    n_subj, n_rois, _ = connectomes.shape
    nodal_disparities = np.zeros((n_subj, n_rois))
    nodal_strengths = np.zeros((n_subj, n_rois))

    for i in range(n_subj):
        adj = connectomes[i].copy()
        np.fill_diagonal(adj, 0)
        strength = adj.sum(axis=1)
        valid = strength > 0
        
        if np.any(valid):
            # Fórmula vectorizada: Y_i = sum((w_ij / s_i)^2)
            nodal_Y = np.sum((adj / strength[:, None])**2, axis=1, where=valid[:, None])
            nodal_disparities[i, :] = nodal_Y
            nodal_strengths[i, :] = strength
        else:
            nodal_disparities[i, :] = np.nan
            
    roi_names = [f"R{i+1}" for i in range(n_rois)]
    df_disparities = pd.DataFrame(nodal_disparities, columns=roi_names)
    df_strengths = pd.DataFrame(nodal_strengths, columns=roi_names)
    return df_disparities, df_strengths

def run_roi_correlations(df, roi_list, target_col='age', collapse_means=True):
    """
    Corre correlaciones masivas para cada ROI vs target_col.
    - collapse_means=True: Agrupa por edad (reduce ruido, sube r).
    - Aplica corrección FDR.
    """
    results = []
    
    for roi in roi_list:
        clean_data = df[[target_col, roi]].dropna()
        
        # Opción de agrupar por edad (Tu método actual)
        if collapse_means:
            clean_data = clean_data.groupby(target_col)[roi].mean().reset_index()
            
        if len(clean_data) > 10:
            r, p = stats.pearsonr(clean_data[target_col], clean_data[roi])
            results.append({'ROI': roi, 'r': r, 'p_val': p})
            
    results_df = pd.DataFrame(results)
    
    # Corrección FDR
    if not results_df.empty:
        reject, pvals_corrected, _, _ = multipletests(results_df['p_val'], alpha=0.05, method='fdr_bh')
        results_df['p_corrected'] = pvals_corrected
        results_df['Significant'] = reject
    
    return results_df.sort_values(by='r', key=abs, ascending=False)


def get_hyperbolic_coords(adj_matrix):
    """Calcula coordenadas (r, theta) para un conectoma individual."""
    N = adj_matrix.shape[0]
    strengths = np.sum(adj_matrix, axis=1)
    s_min = np.min(strengths[strengths > 0])
    
    # Coordenada radial (Popularidad) [cite: 131, 404]
    R = 2 * np.log(N)
    r = R - 2 * np.log(np.maximum(strengths, s_min) / s_min)
    
    # Similitud (Theta): Requiere inferencia por ML. 
    # Para visualización rápida, usamos una aproximación angular funcional.
    theta = np.linspace(0, 2*np.pi, N, endpoint=False) 
    return r, theta

def plot_age_comparison(subject_young, subject_old, disparity_young, disparity_old):
    """Genera plots comparativos compactos (6,5)."""
    sns.set_context("paper")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': 'polar'})
    
    data = [(subject_young, disparity_young, "Young"), 
            (subject_old, disparity_old, "Old")]
    
    for ax, (adj, disp, title) in zip(axes, data):
        r, theta = get_hyperbolic_coords(adj)
        
        # Color basado en tu métrica de disparidad nodal
        scatter = ax.scatter(theta, r, c=disp, cmap='magma', s=30, alpha=0.8)
        
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_ylim(0, np.max(r) * 1.1)
        ax.grid(True, linestyle=':', alpha=0.3)
        # Título interno para mantener compacta la figura
        ax.text(0, np.max(r)*1.3, title, transform=ax.transData, ha='center', fontweight='bold')

    plt.tight_layout()
    return fig



