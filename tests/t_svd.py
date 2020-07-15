    
import numpy as np

pm = np.random.rand(1024, 1400)
pm[10:20] = 0

def svd_inv_fun(poke_mat):        
    dm_eff = (poke_mat**2).sum(axis = 1)
    dm_used = dm_eff > 0.02
    pm_t = poke_mat[dm_used, :]
    U, D, Vt = np.linalg.svd(pm_t, full_matrices=False)
    D2 = 1/(D + 1e-10)
    D2[D < 1e-4] = 0
    Mat = Vt.T @ np.diag(D2) @ U.T
    out = np.hstack([dm_used.astype(np.float32), Mat.flatten().astype(np.float32)])
    return []

svd_inv_fun(pm)

if(type(np.zeros((1,1))) == np.ndarray):
    print('yes')