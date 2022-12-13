# BFTrainer with log Replayer
A scheduler log replayer for BFTrainer evaluation

This is a temporary repository to host source for paper: `CCGRID 2023 submission 2834` which is currently under double-blind review. This repo will be moved to our orginization account once review is done. 

- This simulator replays real scheduler logs for evaluation of different DNN training scenarios and objective metrics. 
- The simulator uses exactly the same MILP implementation as it in the BFTrainer.
- Only the interaction between BFTrainer and main batch scheduler (i.e., main scheduler notifies BFTrainer when there are nodes become idle or preempted) is replaced with a log replayer. 

Target user of the simulator:

- Supercomputer provider: replay their logs to see if BFTrainer can successfully make use of the fragmented idle nodes. 
- BFTrainer users: Try their objective metric and DNN training scenarios (e.g., NAS, HPO) without requesting real resource access.
- Algorithm Developer/Contributor: Evaluate change and proposals without touching the physucal supercomputer and requiring support from the main scheduler.



# Files

- `BFTrainer-replay.py` the main program to replay real scheduler to evaluate our resource allocation algorithm 
- `jobs.py` implements functions to manage jobs.
- `progCBC.py` or `progGRB.py` the implementation of mixed Integer linear programming using Gurobi optimizer (progGRB.py). We also open source our implementation (progGRB.py) using free optimizer (e.g., CBC, Pulp and JuMP). You can get an Trial Licenses or Free Academic Licenses from Gurobi if you want to run the current version. You need to adjust the import source in the `BFTrainer-replay.py` to use the CBC based solver.
- `trace.py` has functions to manage scheduler logs for the replay evaluation.



# Mixed-Integer Linear Programming (MILP) solver

We provided two implementations of the MILP model:
- [PYTHON-MIP](https://www.python-mip.com) with open source (free) [CBC](https://github.com/coin-or/Cbc) solver.
- [gurobipy](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_grbpy_the_gurobi_python.html), the Gurobi Python Interface, with [Gurobi](https://www.gurobi.com) solver. Licensed required, one can use free trial or academia license for evaluation purpose.

Based on our preliminary benchmark, Gurobi is much faster than CBC when problem size is big (e.g., dozens of jobs on hundreds of nodes).
Otherwise, the time to solve is very similar between Gurobi and CBC when problem size is small.<br>

The PYTHON-MIP also supports using Gurobi as solver (license required as well) but slower than gurobipy in most cases especially when problem size is large.
Thus, one can use the free CBC when problem size is small, especailly for relatively small supercomputers. 
Otherwise, Gurobi with the gurobipy based implementation is recommended.

## Gurobi 

Our implementation requires Gurobi to solve the MILP algorithm. 
### Installation 
you can install the Gurobi python package using `python -m pip install gurobipy` or other ways shown here: https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_python_installation_opt.html

### License 
You can either get a free trial liecense or academic liecense from https://www.gurobi.com/academia/academic-program-and-licenses/.
Once you get the license, you need to set the environment variable **GRB_LICENSE_FILE** to the full path of the license file.
You can set the environment variable by editing line 8 of `BFTrainer-replay.py`

## PYTHON-MIP
Please refer to the [offical site](https://python-mip.readthedocs.io/en/latest/install.html) for installation instrcution. Basically, you only need `pip install mip` and it comes with CBC solver.



# Inputs

## Tasks
As discussed in the section 3 of the paper. Some parameters, such as the minimum and maximum number of nodes each task can run on, must be supplied by user when submit their tasks to BFTrainer. Currently we dumped a pandas dataframe that containes all the information about the tasks supplied by user into python pickle file. One can see an example at `./dataset/hps-shufflenet-Jbnd-10J.pkl`. The file can be supplied using argument `-bdfn`.
Similiarly, an example of the scalability information can be found at `./dataset/hps-shufflenet-JNPs-10J.pkl`, thi is optional in practice (BFTrainer will need to benchmark the tasks if not supplied) but needed for simulation. This file can be supplied using argument `-scfn`.

## Replaying Logs
In practice, BFTrainer talks with the main batch scheduler to get the status change of the resource pool, i.e., BFTrainer is notified (called event in the paper) when any nodes are released/allocated from/for jobs in the main scheduler.
For simulation, we stored the series of timestamped events in a file, an example can be fond at `./dataset/ndstat-10sdn6-evt.pkl` (please extract it from `ndstat-10sdn6-evt.tar.gz`) and must be supplied using `-ndmap`.

## Forward looking time
This is the only parameters that an admin or user need to configure based on the characteristics of the idle nodes of the target supercomputer. 
It can be supplied using argument `-Tfwd`.
