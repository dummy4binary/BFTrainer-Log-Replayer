import numpy as np
import sys, os, time, argparse, h5py, shutil
from jobs import gen_rand_job_map, interp1d4s, load_job_cfg, synthetic_jobs, rescale_cost2outcome
from trace import deduplicate_events
import pandas as pd 
from util import str2bool

os.environ['GRB_LICENSE_FILE'] = '/path/to/gurobi.lic'
import gurobipy as grb
from progGRB import re_allocate, re_allocate_ndf

parser = argparse.ArgumentParser(description='MIP for optimal resource allocation, one step only')
parser.add_argument('-nis',   type=int, default=0, help='starting node index')
parser.add_argument('-nN',    type=int, default=None, help='first N nodes')
parser.add_argument('-Tfwd',  type=int, default=10, help='time of expected forward')
parser.add_argument('-optTL', type=int, default=10, help='time limit for optimizer')
parser.add_argument('-name',  type=str, default='debug', help='evaluation name')
parser.add_argument('-ndmap', type=str, default='./dataset/ndstat-10sdn6-evt.pkl', help='node state map data')
parser.add_argument('-bdfn',  type=str, default='./dataset/hps-shufflenet-Jbnd-10J.pkl', help='job bound info')
parser.add_argument('-scfn',  type=str, default='./dataset/hps-shufflenet-JNPs-10J.pkl', help='job scalability')
parser.add_argument('-print', type=str2bool, default=True, help='1: print to terminal; 0: redirect to file')
parser.add_argument('-mip',   type=str2bool, default=True, help='1: use mip; 0: use heuristic')
parser.add_argument('-maxJ',  type=int, default=10, help='max parallel jobs')

args, unparsed = parser.parse_known_args()
if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

out_dir = args.name + '-itrOut'
if os.path.isdir(out_dir): 
    shutil.rmtree(out_dir)
os.mkdir(out_dir) # to save temp output

# redirect print to a file
if not args.print:
    sys.stdout = open('%s/%s' % (out_dir, 'iter-prints.log'), 'w') 

# Jbnd, JNPs = synthetic_jobs(20, tsub_range=(0, 5))
Jbnd, JNPs = load_job_cfg(args.bdfn, args.scfn)

nstat = pd.read_pickle(args.ndmap).iloc[1000:, args.nis:(args.nis+args.nN)] # 0:busy, 1:idle
nstat = deduplicate_events(nstat) # remove duplicated events because of subsetting nodes

qstat = pd.Series(np.zeros(Jbnd.shape[0], dtype=np.int8), index=Jbnd.index) # 1:runing/completed 0:queued
nd_id = nstat.columns

_jmask = Jbnd.Tsub <= nstat.index[0]
if args.maxJ < _jmask.sum():
    _mask_ready = np.nonzero(_jmask.values)[0]
    _jmask.iloc[_mask_ready[args.maxJ:]] = False
QJbnd = Jbnd[_jmask].copy()
QJNPs = JNPs[_jmask].copy()
qstat[_jmask] = 1

init_jmap = gen_rand_job_map(nJ=QJbnd.shape[0], nN=nstat.iloc[0].sum(), \
                             jmin=QJbnd.Jmin, jmax=QJbnd.Jmax)
init_nds  = nd_id[nstat.iloc[0].values == 1]
cmap = pd.DataFrame(init_jmap, columns=init_nds, index=QJbnd.index)
# cmap.to_csv('%s/%05d-initial.csv' % (out_dir, 0))
if QJNPs.shape[0] > 0: print('[Start] %s %s' % (nstat.index[0], ', '.join(QJNPs.index)))

_nJ = cmap.sum(axis=1)
job_rate = pd.Series(np.array([interp1d4s(QJNPs.Ns[_j], QJNPs.Ps[_j], _nJ[_j]) if _nJ[_j] >= QJbnd.Jmin[_j] else 0 for _j in cmap.index]), \
                     index=QJbnd.index)
job_accs = pd.Series(np.zeros(QJbnd.shape[0]), index=QJbnd.index)
rescale_cost = job_rate * QJbnd.Tcup # initial start up cost 

fp_step_log = open("%s/step-logs.csv" % out_dir, 'w')
fp_step_log.write('evtTime,Thr,Rate,ReScost,evt,Nleft,Npre,Njoin,nJob,nNd,nJrun,nNrun,PreCost\n')
for _idx in range(1, nstat.shape[0], 1):        
    _telps = (nstat.index[_idx] - nstat.index[_idx-1]).total_seconds()
    _progr = pd.Series(_telps * job_rate, index=QJbnd.index) - rescale_cost
    _pstat = nstat.iloc[_idx-1].values
    _cstat = nstat.iloc[_idx].values
    
    _clr_nds = nd_id[(_pstat==1) & (_cstat==0)] # nodes left in between 
    _cnt_left= _clr_nds.shape[0]

    prempt_cost  = pd.Series(np.zeros(QJbnd.shape[0]), index=QJbnd.index)
    for _ni in _clr_nds:
        if cmap[_ni].sum() == 1: # if node is used
            _jn = cmap[_ni].idxmax()
            # to make sure only penalize once even if multiple nodes left
            if prempt_cost[_jn] == 0: 
                # those jobs got half progress(estimate)
                prempt_cost[_jn] = QJbnd.Tcdw[_jn] * job_rate[_jn]
                _progr[_jn]     -= prempt_cost[_jn] 
        cmap.drop(_ni, axis=1, inplace=True) # remove nodes from the map
    
    nJ_preempted = prempt_cost[prempt_cost>0].shape[0]
    job_accs += _progr
    # remove finished job from decision
    _js2rm = []
    for _j in cmap.index:
        if QJbnd.Pmax[_j] <= job_accs[_j]: # terminate job as it has completed
            _js2rm.append(_j)
            _progr[_j]  -= job_accs[_j] - QJbnd.Pmax[_j]
            job_accs[_j] = QJbnd.Pmax[_j]

    cmap.drop(_js2rm,  axis=0, inplace=True)
    QJbnd.drop(_js2rm, axis=0, inplace=True)
    QJNPs.drop(_js2rm, axis=0, inplace=True)
    job_accs.drop(_js2rm, inplace=True)
    if len(_js2rm) > 0: print('[Done] %s %s' % (nstat.index[_idx], ', '.join(_js2rm)))

    _add_nds = nd_id[(_pstat==0) & (_cstat==1)] # new nodes joined
    _cnt_join= _add_nds.shape[0]
    for _ni in _add_nds:
        cmap[_ni] = np.zeros(cmap.shape[0], dtype=np.int8)
        
    # added newly submitted jobs into consideration
    if QJbnd.shape[0] < args.maxJ: 
        _jmask = (Jbnd.Tsub <= nstat.index[_idx]) & (qstat == 0) # can be all False!
        if args.maxJ < _jmask.sum() + QJbnd.shape[0]:
            _mask_ready = np.nonzero(_jmask.values)[0]
            _jmask.iloc[_mask_ready[(args.maxJ-QJbnd.shape[0]):]] = False
        _new_subJbnd = Jbnd[_jmask].copy()
        _new_subJNPs = JNPs[_jmask].copy()
        qstat[_jmask] = 1
    else:
        _new_subJbnd, _new_subJNPs = None, None

    if _new_subJbnd is not None: 
        QJbnd = QJbnd.append(_new_subJbnd, verify_integrity=True)
        QJNPs = QJNPs.append(_new_subJNPs, verify_integrity=True)
        cmap  = cmap.append(pd.DataFrame(np.zeros((_new_subJbnd.index.shape[0], cmap.shape[1]), dtype=np.int8), \
                                         index=_new_subJbnd.index, columns=cmap.columns), \
                            verify_integrity=True)
        for _j in _new_subJbnd.index: job_accs[_j] = 0
        if _new_subJbnd.shape[0] > 0: print('[Start] %s %s' % (nstat.index[_idx], ', '.join(_new_subJbnd.index)))

    # current Queue finished, and no sub in the future (replay)
    if cmap.shape[0]==0 and qstat.sum()==qstat.shape[0]:
        print('there is no more job to run')
        fp_step_log.write("%s,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d,%d,%f\n" % (nstat.index[_idx], sum(_progr), job_rate.sum(), 0, _idx, _cnt_left,\
                          nJ_preempted, _cnt_join, 0, cmap.shape[1], 0, 0, prempt_cost.sum()))
        break
    else:
        if cmap.shape[0]>0: 
            print('[progress]; ' + '; '.join(['%s : %.2f%% = %.2f / %.2f' % (_jn, \
                  100*job_accs[_jn]/QJbnd.Pmax[_jn], job_accs[_jn], QJbnd.Pmax[_jn]) \
                  for _jn in job_accs.index]))

    # cmap.to_csv('%s/%05d-initial.csv' % (out_dir, _idx))

    if cmap.shape[1] == 0: # no node to use
        print('[WARN] there is no node to use by scavenger', cmap.shape)
        rescale_cost = np.zeros(cmap.shape[0])
        job_rate = pd.Series(np.zeros(cmap.shape[0]), index=QJbnd.index)
        _nJ, _nA = cmap.sum(axis=1), cmap.sum(axis=0)
        fp_step_log.write("%s,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d,%d,%f\n" % (nstat.index[_idx], sum(_progr), job_rate.sum(), rescale_cost.sum(), _idx, _cnt_left,\
                          nJ_preempted, _cnt_join, cmap.shape[0], cmap.shape[1], _nJ[_nJ > 0].shape[0], _nA[_nA > .9].shape[0], prempt_cost.sum()))
        continue

    if cmap.shape[0] == 0: # empty queue
        print('[WARN] there is no job to run by scavenger, but expect jobs to come later', cmap.shape)
        rescale_cost = np.array([])
        job_rate = pd.Series(np.array([]), index=QJbnd.index)
        _n, _nA = cmap.sum(axis=1), cmap.sum(axis=0)
        fp_step_log.write("%s,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d,%d,%f\n" % (nstat.index[_idx], sum(_progr), job_rate.sum(), rescale_cost.sum(), _idx, _cnt_left,\
                          nJ_preempted, _cnt_join, cmap.shape[0], cmap.shape[1], _nJ[_nJ > 0].shape[0], _nA[_nA > .9].shape[0], prempt_cost.sum()))
        continue

    _jmin = QJbnd.Jmin.values.tolist()
    _jmax = QJbnd.Jmax.values.tolist()
    _cup  = QJbnd.Tcup.values.tolist()
    _cdw  = QJbnd.Tcdw.values.tolist()
    _Ns   = QJNPs.Ns.values.tolist()
    _Ps   = QJNPs.Ps.values.tolist()
    _Os   = QJNPs.Os.values.tolist()

    if args.mip:
        opt_status, sol_map, credit_rate, rescale_obj_cost = \
            re_allocate(cmap=cmap.values, jmin=_jmin, jmax=_jmax, Ns=_Ns, Os=_Os, Tfwd=args.Tfwd, \
                        res_up=_cup, res_dw=_cdw, time_limit=args.optTL)
    else:
        opt_status, sol_map, credit_rate, rescale_obj_cost = \
            re_allocate_ndf(cmap=cmap.values, jmin=_jmin, jmax=_jmax, Ns=_Ns, Os=_Os, \
                            res_up=_cup, res_dw=_cdw)

    if opt_status == grb.GRB.OPTIMAL:
        rescale_cost = rescale_cost2outcome(cmap.values, sol_map, _Ns, _Ps, _cup, _cdw)
        cmap = pd.DataFrame(sol_map, columns=cmap.columns, index=QJbnd.index)
    else:
        print('[WARN] %s Gurobi TIME-LIMIT when allocating %d nodes for %d jobs, %d nodes used' % (\
              nstat.index[_idx], cmap.shape[1], cmap.shape[0], cmap.values.sum()))
        rescale_cost = np.zeros(cmap.shape[0])

    _nJ = cmap.sum(axis=1)
    job_rate = pd.Series(np.array([interp1d4s(_Ns[_j], _Ps[_j], _nJ[_j]) if _nJ[_j] >= _jmin[_j] else 0 for _j in range(cmap.shape[0])]),\
                        index=QJbnd.index)

    # cmap.to_csv('%s/%05d-reprogrammed.csv' % (out_dir, _idx))

    _nJ = cmap.sum(axis=1)
    _nA = cmap.sum(axis=0)
    fp_step_log.write("%s,%f,%f,%f,%d,%d,%d,%d,%d,%d,%d,%d,%f\n" % (nstat.index[_idx], sum(_progr), job_rate.sum(), rescale_cost.sum(), _idx, _cnt_left,\
                      nJ_preempted, _cnt_join, cmap.shape[0], cmap.shape[1], _nJ[_nJ > 0].shape[0], _nA[_nA > .9].shape[0], prempt_cost.sum()))
    sys.stdout.flush()
    fp_step_log.flush()

sys.stdout.flush()
fp_step_log.close()
