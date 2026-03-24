
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import networkx as nx
import bct

import bct.algorithms.modularity as bct_mod

from joblib import Parallel, delayed

def safe_ls2ci(ls, zeroindexed=False):
    """
    Versión parcheada de ls2ci que evita el error de ragged arrays de NumPy.
    Usa len(ls) == 0 en lugar de np.size(ls) == 0.
    """
    if ls is None or len(ls) == 0:
        return ()
    nr_indices = sum(map(len, ls))
    ci = np.zeros((nr_indices,), dtype=int)
    for i, mod in enumerate(ls):
        for node in mod:
            ci[node] = i if zeroindexed else i + 1
    return ci

# Aplicamos el parche a la librería en memoria
bct_mod.ls2ci = safe_ls2ci
bct.ls2ci = safe_ls2ci

def _bct_worker_exact(adj_cd):
    """Worker replicating the exact MATLAB BCT formulas."""
    adj = adj_cd.copy()
    np.fill_diagonal(adj, 0)
    
    # 1. Clustering
    clustering = np.mean(bct.clustering_coef_wu(adj))
    
    # 2. Modularity (Author uses default gamma=1.0)
    _, modularity = bct.modularity_und(adj, gamma=1.0)
    
    # 3. Path Length & Global Efficiency (Author uses binary topological distance). !! 
    D = bct.distance_bin(adj)
    charpath_res = bct.charpath(D, 0, 0)
    
    return clustering, modularity, charpath_res[0], charpath_res[1], charpath_res[4]

def calculate_graph_metrics(connectomes_cd, connectomes_vd):
    n_subj = connectomes_cd.shape[0]
    
    # Vectorized Strength (Instanteous)
    row_sums = np.sum(connectomes_vd, axis=2)
    diagonals = np.diagonal(connectomes_vd, axis1=1, axis2=2)
    strengths = np.mean(row_sums - diagonals, axis=1)
    
    # Disparity (Assuming you have this vectorized or fast)
    disparities = np.array([calculate_disparity(connectomes_vd[i]) for i in range(n_subj)])
    disparities_cd = np.array([calculate_disparity(connectomes_cd[i]) for i in range(n_subj)])
    
    # Parallel BCT execution
    results = Parallel(n_jobs=-1)(
        delayed(_bct_worker_exact)(connectomes_cd[i]) for i in range(n_subj)
    )
    
    clustering_coeffs, modularities, avg_shortest_paths, global_efficiencies, diameters = zip(*results)
    
    #Repeat strength but with the connectomes_cd for reproducibility
    
    strengths_cd = np.mean(np.sum(connectomes_cd, axis=2) - np.diagonal(connectomes_cd, axis1=1, axis2=2), axis=1)
    
    #Calculate density by comparing to a fully connected graph of 90 nodes (without self-loops)
    density = np.array([np.sum(connectomes_vd[i] > 0) / (90 * 89) for i in range(n_subj)])
    
    return pd.DataFrame({
        'Strength': strengths, 
        'Strength_CD': strengths_cd,
        'Disparity_Y': disparities,
        'Disparity_CD': disparities_cd,
        'Clustering_Coefficient': clustering_coeffs,
        'Average_Shortest_Path': avg_shortest_paths,
        'Global_Efficiency': global_efficiencies,
        'Diameter': diameters,
        'Modularity': modularities,
        'Density': density
    })
    
    

def calculate_disparity(adj):
    strength = adj.sum(axis=1)
    valid = strength > 0    
    if np.any(valid):
        nodal_Y = np.sum((adj[valid] / strength[valid, None])**2, axis=1)
        disparity = nodal_Y.mean()
    else:
        disparity = 0
    return disparity


def calculate_disparities(connectomes):

    # Preparamos array para guardar la media de Y por sujeto
    disparities = np.zeros(connectomes.shape[0])
    
    for i in range(connectomes.shape[0]):
        # Coger matriz del sujeto i
        adj = connectomes[i]
        
        # Asegurar diagonal a 0 (self-loops)
        np.fill_diagonal(adj, 0)
        
        # Calcular Strength (suma de pesos por nodo)
        strength = adj.sum(axis=1)
        
        # Evitar división por cero (nodos desconectados)
        valid = strength > 0
        
        # Fórmula de Disparidad Nodal: Y_i = sum((w_ij / s_i)^2)
        # (Suma de los cuadrados de los pesos normalizados)
        if np.any(valid):
            # Vectorizamos el cálculo para las 90 regiones
            nodal_Y = np.sum((adj[valid] / strength[valid, None])**2, axis=1)
            disparities[i] = nodal_Y.mean()
        else:
            disparities[i] = 0 # Caso raro: cerebro desconectado
    print(f"Global Average Disparity: {disparities.mean():.4f}")
    return disparities



def get_epoch_correlations(df, metric, epochs):
    results = []
    for label, (start, end) in epochs.items():
        window = df[(df['age'] >= start) & (df['age'] <= end)].dropna(subset=['age', metric])
        # We take [:, 0] in case there are duplicated columns, taking only the first one
        x = np.array(window['age']).reshape(-1, 1)[:, 0].astype(float)
        y = np.array(window[metric]).reshape(-1, 1)[:, 0].astype(float)
        
        if len(x) > 2:
            r, p = stats.pearsonr(x, y)
            results.append({
                'Metric': metric, 
                'Epoch': label, 
                'r': round(float(r), 4), 
                'p-value': float(p)
            })

    return pd.DataFrame(results)

def run_native_r_gam_exact(df, metric_col):
    df = df.dropna().copy()
    with (ro.default_converter + pandas2ri.converter).context():
        ro.globalenv['r_df'] = ro.conversion.get_conversion().py2rpy(df)
    
    # Literal translation of the author's exact R string (NO as.factor).
    # Actually I should use r_df$dataset <- as.factor(r_df$dataset) for dataset and atlas as they are categorical varibales.
    r_code = f"""
    library(mgcv)
    model <- gam({metric_col} ~ s(age, bs="cr") + sex + dataset + atlas, data=r_df, method='REML')
    summary(model)
    """
    return ro.r(r_code)

from pymatreader import read_mat

def load_alexa_graph_metrics(demo, mat_data):
    
    
    # The R script specifies the matrix key is 'organizational_measures'
    measures_array = mat_data['organizational_measures'] 
    
    # The R script maps these exact 12 columns in this order
    col_names = [
        'global_efficiency', 'path_length', 'small_worldness', 'strength', 
        'modularity', 'core_periphery', 'kcore', 'score', 'local_efficiency', 
        'clustering', 'betweenness', 'subgraph_centrality'
    ]
    
    # 3. Create metrics dataframe
    measures_df = pd.DataFrame(measures_array, columns=col_names)
    measures_df = measures_df.reset_index(drop=True)
    
    # 4. Positional concatenation (strictly mimics R's bind_cols)
    global_data = pd.concat([demo, measures_df], axis=1)
    
    return global_data


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



