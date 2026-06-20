
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
 
 
# ── Colores ──────────────────────────────────────────────────────────────────
 
def make_ds_colors(ds_list):
    """Devuelve dict {dataset_id: color} con tab10."""
    cmap = plt.cm.tab10
    return {ds: cmap(i / max(len(ds_list) - 1, 1)) for i, ds in enumerate(ds_list)}
 
 
# ── Métricas de red ──────────────────────────────────────────────────────────
 
def node_metrics(adj):
    """
    Métricas por nodo para una matriz de adyacencia pesada.
    Devuelve: k (degree), s (strength), Y (disparity), kY (k·Y)
    """
    A = adj.copy(); np.fill_diagonal(A, 0)
    k  = (A > 0).sum(axis=1).astype(float)
    s  = A.sum(axis=1)
    p  = A / np.where(s > 0, s, 1.)[:, None]
    Yi = (p**2).sum(axis=1)
    #average weight of a node
    wi = np.where(k > 0, s / k, 0)
    
    return k, s, Yi, k * Yi, wi
 
 
def subject_metrics(conn_array, N_roi=90):
    """
    Métricas escalares por sujeto para un array de conectomas.
    Devuelve DataFrame con: density, strength, degree, Y_obs, kY_obs, Y_null, Ratio
    """
    n = len(conn_array)
    D, S, K, Y, kY, YN, R = [np.zeros(n) for _ in range(7)]
    for i, raw in enumerate(conn_array):
        adj = raw.copy(); np.fill_diagonal(adj, 0)
        k, s, Yi, kYi = node_metrics(adj)
        v = k > 1
        if v.sum() == 0:
            D[i] = S[i] = K[i] = Y[i] = kY[i] = YN[i] = R[i] = np.nan
            continue
        D[i]  = (adj > 0).sum() / (N_roi * (N_roi - 1))
        S[i]  = s[v].mean()
        K[i]  = k[v].mean()
        Y[i]  = Yi[v].mean()
        kY[i] = kYi[v].mean()
        YN[i] = (2. / (k[v] + 1)).mean()
        R[i]  = (Yi[v] / (2. / (k[v] + 1))).mean()
    return pd.DataFrame(dict(density=D, strength=S, degree=K,
                             Y_obs=Y, kY_obs=kY, Y_null=YN, Ratio=R))
 
 
def subj_stats(adj):
    """Stats de red (min/max/mu/var de k, w, s) para un conectoma. Devuelve dict."""
    A = adj.copy(); np.fill_diagonal(A, 0)
    k = (A > 0).sum(axis=1).astype(float)
    s = A.sum(axis=1); w = A[A > 0]; v = k > 0
    if v.sum() == 0 or len(w) == 0:
        return {}
    return {'N': int(v.sum()),
            'k_min': k[v].min(), 'k_max': k[v].max(),
            'k_mu': k[v].mean(), 'k_var': k[v].var(),
            'w_min': w.min(),    'w_max': w.max(),
            'w_mu':  w.mean(),   'w_var': w.var(),
            's_min': s[v].min(), 's_max': s[v].max(),
            's_mu':  s[v].mean(),'s_var': s[v].var()}
 
 
def by_degree(conn_list):
    """
    Para cada clase de degree k, calcula media de Y, kY y Ratio
    sobre todos los conectomas de conn_list.
    Devuelve dict {k: (Y_mean, Ratio_mean, kY_mean, n_connectomes)}
    """
    acc = defaultdict(lambda: {'Y': [], 'R': [], 'kY': [], 'nc': 0})
    for adj in conn_list:
        k, s, Yi, kYi = node_metrics(adj)
        seen = set()
        for ki in np.unique(k[k > 1].astype(int)):
            m = k == ki
            acc[ki]['Y'].append(Yi[m].mean())
            acc[ki]['kY'].append(kYi[m].mean())   # ← fix: era kYi[i]
            acc[ki]['R'].append((Yi[m] / (2. / (ki + 1))).mean())
            seen.add(ki)
        for ki in seen:
            acc[ki]['nc'] += 1
    return {ki: (np.mean(v['Y']), np.mean(v['R']), np.mean(v['kY']), v['nc'])
            for ki, v in acc.items()}
 
 
def grep_YR(gr):
    """Y_obs y Ratio para un group representative (matriz 90×90)."""
    k, s, Yi, kYi = node_metrics(gr)
    v = k > 1
    if v.sum() == 0:
        return np.nan, np.nan
    Yn = 2. / (k[v] + 1)
    return Yi[v].mean(), (Yi[v] / Yn).mean()
 
 
# ── Null model kY(k) references (Serrano et al. 2009 eq. 5-6) ────────────────
 
def null_model_kY(ks=None):
    """
    Devuelve arrays de referencia para el scatter kY(k):
      kY_null (media), kY_null_hi (media+2σ), kY_null_lo (=1),
      kY_hom (totalmente homogéneo), kY_het (totalmente heterogéneo)
    """
    if ks is None:
        ks = np.arange(2, 90)
    kY_null = 2 * ks / (ks + 1)
    term1   = (20 + 4*ks) / ((ks+1) * (ks+2) * (ks+3))
    term2   = 4 / (ks+1)**2
    var_kY  = (ks**2) * (term1 - term2)
    return dict(
        ks      = ks,
        kY_null = kY_null,
        kY_hi   = kY_null + 2*np.sqrt(var_kY),
        kY_lo   = np.ones_like(ks, dtype=float),
        kY_hom  = np.ones_like(ks, dtype=float),
        kY_het  = ks.astype(float),
    )
 