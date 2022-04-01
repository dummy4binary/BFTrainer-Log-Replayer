import numpy as np
import pandas as pd 

def Amdahl_law(n, p=.98):
    return 1 / (1 - p + p / n)
    
def gen_rand_job_map(nJ, nN, jmin, jmax):
    assert (nJ == len(jmin) and nJ == len(jmax)), 'aug error'
    if nN == 0: return np.zeros((nJ, 0), dtype=np.int8)
    # a random current state
    tmp_map = []
    for _n in range(nN):
        _j = np.random.randint(0, nJ)
        tmp_map.append(np.eye(nJ)[_j].tolist())

    # J x N
    c_map = np.array(tmp_map).T

    for _j in range(nJ):
        _ndj = c_map[_j].sum()
        if _ndj >= jmin[_j] and _ndj <= jmax[_j]:continue
        if _ndj < jmin[_j]: c_map[_j] *= 0
        clear_cnt = 0
        for _n in range(nN):
            if c_map[_j][_n]:
                c_map[_j][_n] = 0
                clear_cnt += 1

            if clear_cnt == _ndj - jmax[_j]:
                break

    return c_map.astype(np.int8)

def interp1d4s(Ns, Ps, Nx):
    from scipy import interpolate
    f = interpolate.interp1d(Ns, Ps)
    return f(Nx)

def load_job_cfg(bd_fn, sc_fn):
    Jbnd = pd.read_pickle(bd_fn)
    JNPs = pd.read_pickle(sc_fn)
    return Jbnd, JNPs

def synthetic_jobs(nJ, tsub_range, rnd_seed=2021):
    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    Jmin = [4] * nJ
    Jmax = [128] * nJ

    _basec = np.random.randint(5, 20, nJ)
    Tcup = _basec * 2
    Tcdw = _basec * 1

    Ns = []
    Ps = []

    S_j = (4 * np.random.randint(1, 64, nJ)).tolist()
    for _j in range(nJ):
        _nlb, _nub = Jmin[_j], Jmax[_j]
        _Ns = list(range(_nlb, _nub+1, 1))
        _p = [S_j[_j] / _nlb * Amdahl_law(_n) for _n in _Ns]
    #     _p = [S_j[_j] / _nlb * _n for _n in _Ns]
        Ns.append([0, ] + _Ns)
        Ps.append([0, ] + _p)
    Os = Ps
    Pmax = [np.random.randint(100, 200)*_P[-1] for _P in Ps]
    Tsub = np.linspace(tsub_range[0], tsub_range[1], nJ//2)
    Tsub = np.hstack([np.zeros(nJ-Tsub.shape[0]), Tsub])
    
    Jname= ['J%05d' % _j for _j in range(nJ)]
    Jbnd = pd.DataFrame(np.vstack([Tcup, Tcdw, Jmin, Jmax, Pmax, Tsub]).T, \
                        columns=('Tcup', 'Tcdw', 'Jmin', 'Jmax', 'Pmax', 'Tsub'),\
                        index=Jname)
    JNPs = pd.DataFrame([(_n, _p, _o) for _n, _p in zip(Ns, Ps, Os)], columns=['Ns', 'Ps', 'Os'], index=Jname)
    JNPs['Tsub'] = Tsub
    return Jbnd, JNPs

def rescale_cost2outcome(pmap, cmap, Ns, Ps, Tcup, Tcdw):
    nJ  = pmap.shape[0]
    ret = np.zeros(nJ)
    p_nd = pmap.sum(axis=1)
    c_nd = cmap.sum(axis=1)
    for _j in range(nJ):
        nd_dw = ((pmap[_j]==1) & (cmap[_j]==0)).sum()
        nd_up = ((pmap[_j]==0) & (cmap[_j]==1)).sum()
        if nd_dw > 0 and nd_up > 0: # should never arrive only if MIP is wrong
            print('[WARN] job %dth has node migration, Up%d, Down%d' % (_j, nd_up, nd_dw))
            continue
        if nd_dw > 0:
            ret[_j] = interp1d4s(Ns[_j], Ps[_j], p_nd[_j]) * Tcdw[_j]
            # print('down-scaled job %d for %d nodes from %d to %d costs %.1f imgs' % (_j, nd_dw, p_nd[_j], c_nd[_j], ret[_j]))
        elif nd_up > 0:
            ret[_j] = interp1d4s(Ns[_j], Ps[_j], p_nd[_j]) * Tcup[_j]
            # print('up-scaled job %d for %d nodes from %d to %d costs %.1f imgs' % (_j, nd_up, p_nd[_j], c_nd[_j], ret[_j]))
        else:
            ret[_j] = 0
    return ret
