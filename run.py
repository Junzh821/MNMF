import sys
import itertools
import subprocess
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shutil import rmtree
from os import environ, mkdir, path
import tabulate_results


get_results_only = False

switch_gpus = False #For multiple GPUs
n_parallel_threads = 25

# Set Hyper-parameters
args = dict()
# The names should be the same as argument names in parser.py
args['hyper_params'] = ['DATA_DIR', 'ETA', 'ALPHA', 'BETA', 'THETA', 'PHI', 'LAMBDA','L_COMPONENTS', 'K', 'COST_F']
custom = '_MNMF_'
now = datetime.now()
args['timestamp'] = str(now.month)+'|'+str(now.day)+'|'+str(now.hour)+':'+str(now.minute)+':'+str(now.second) + custom #  '05|12|03:41:02'  # Month | Day | hours | minutes (24 hour clock)

args['DATA_DIR'] = ['washington']# ,'washington', 'wisconsin', 'texas', 'cornell', 'ppi', 'wiki_n2v', 'cora', 'citeseer', 'wiki', 'armherst', 'hamilton', 'mich', 'rochester', 'blogcatalog']
args['ETA'] = [1.0]#, 1.0, 5.0]
args['ALPHA'] = [0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0]
args['BETA'] = [0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0]
args['THETA'] = [1.0] #[0.5, 1.0, 5.0, 10.0]
args['PHI'] = [1.0] #[0.5, 1.0, 5.0, 10.0]
args['LAMBDA'] = [1.0] #[0.001, 0.01, 0.1, 1.0]
args['COST_F'] = [5e-5]#, 5e-4, 5e-3]
args['L_COMPONENTS'] = [128]#, 175]
args['K'] = [11]#, 10, 16, 24]

pos = args['hyper_params'].index('DATA_DIR')
args['hyper_params'][0], args['hyper_params'][pos] = args['hyper_params'][pos], args['hyper_params'][0]
#print(args)


if not get_results_only:
    def diff(t_a, t_b):
        t_diff = relativedelta(t_a, t_b)
        return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

    # Create Args Directory to save arguments
    args_path = 'args'
    if not path.exists(args_path):
        mkdir(args_path)
    np.save(path.join('args', args['timestamp']), args)

    #Create Log Directory for stdout Dumps
    stdout_dump_path = 'Results'
    if not path.exists(stdout_dump_path ):
        mkdir(stdout_dump_path)

    param_values = []
    this_module = sys.modules[__name__]
    for hp_name in args['hyper_params']:
        param_values.append(args[hp_name])
    combinations = list(itertools.product(*param_values))
    n_combinations = len(combinations)
    print('Total no of experiments: ', n_combinations)

    pids = [None] * n_combinations
    f = [None] * n_combinations
    last_process = False
    for i, setting in enumerate(combinations):
        #Create command
        command = "python main_algo.py "
        folder_suffix = "L"#args['timestamp']
        for name, value in zip(args['hyper_params'], setting):
            command += "--" + name + " " + str(value) + " "
            folder_suffix += "_"+str(value)
        fldr = path.join(stdout_dump_path, folder_suffix)
        if not path.exists(fldr):
            mkdir(fldr)
        name = path.join(fldr, folder_suffix)
        command += "--" + "LOG_DIR " + name
        print(i+1, '/', n_combinations, command)

        if switch_gpus and (i % 2) == 0:
            env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"})
        else:
            env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"})

        with open(name, 'w') as f[i]:
            pids[i] = subprocess.Popen(command.split(), env=env, stdout=f[i])
        if i == n_combinations-1:
            last_process = True
        if ((i+1) % n_parallel_threads == 0 and i >= n_parallel_threads-1) or last_process:
            if last_process and not ((i+1) % n_parallel_threads) == 0:
                n_parallel_threads = (i+1) % n_parallel_threads
            start = datetime.now()
            print('########## Waiting #############')
            for t in range(n_parallel_threads-1, -1, -1):
                pids[i-t].wait()
            end = datetime.now()
            print('########## Waiting Over######### Took', diff(end, start), 'for', n_parallel_threads, 'threads')

        # Tabulate results in xls
        #tabulate_results.write_results(args)

else:
    #tabulate_results.write_results(args)
    print("Done tabulation")

