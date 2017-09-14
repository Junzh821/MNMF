import sys
import itertools
import subprocess
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shutil import rmtree
from os import environ, mkdir, path, makedirs

get_results_only = False
switch_gpus = False #For multiple GPUs
n_parallel_threads = 1
# Set Hyper-parameters
args = dict()
# The names should be the same as argument names in parser.py
args['hyper_params'] = ['DATA_DIR', 'ETA', 'ALPHA', 'BETA', 'THETA', 'GAMMA', 'LAMBDA', 'K', 'MAX_ITER']
custom = '_MNF_'
now = datetime.now()
args['timestamp'] = str(now.month)+'|'+str(now.day)+'|'+str(now.hour)+':'+str(now.minute)+':'+str(now.second) + custom #  '05|12|03:41:02'  # Month | Day | hours | minutes (24 hour clock)

args['DATA_DIR'] = ['citeseer'] #'washington', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'wiki', 'ppi', 'wiki_n2v', 'blogcatalog', 'armherst', 'hamilton', 'mich', 'rochester']
args['ETA'] = [1.0]
args['ALPHA'] = [0.5, 1.0, 2.0, 5.0]
args['BETA'] = [0.5, 1.0, 2.0, 5.0]
args['THETA'] = [0.5, 1.0, 2.0, 5.0]
args['GAMMA'] = [0.5, 1.0, 2.0, 5.0]
args['LAMBDA'] = [1.0]
args['K'] = [4, 6, 8, 10, 12, 14, 15]
args['MAX_ITER'] = [50, 100, 150]

def run_algo() :
    global get_results_only
    global switch_gpus
    global n_parallel_threads
    global args
    global custom
    global now
    pos = args['hyper_params'].index('DATA_DIR')
    args['hyper_params'][0], args['hyper_params'][pos] = args['hyper_params'][pos], args['hyper_params'][0]
    if not get_results_only:
        def diff(t_a, t_b):
            t_diff = relativedelta(t_a, t_b)
            return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)

        # Create Args Directory to save arguments
        args_path = 'args'
        if not path.exists(args_path):
            mkdir(args_path)
        np.save(path.join('args', args['timestamp']), args)

        # Create Log Directory for stdout Dumps
        stdout_dump_path = 'emb'
        if not path.exists(stdout_dump_path):
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
            # Create command
            command = "python main_algo.py "
            folder_suffix = "MNF"
            dataset_name = ''
            for name, value in zip(args['hyper_params'], setting):
                command += "--" + name + " " + str(value) + " "
                folder_suffix += "_" + str(value)
                if name == 'DATA_DIR':
                    dataset_name = value
            fldr = path.join(stdout_dump_path, dataset_name.title(), folder_suffix)
            if not path.exists(fldr):
                makedirs(fldr, exist_ok=True)
            name = path.join(fldr, folder_suffix)
            command += "--" + "LOG_DIR " + fldr + " " +"--FOLDER_SUFFIX" + " " + folder_suffix + " "
            print(i + 1, '/', n_combinations, command)

            if switch_gpus and (i % 2) == 0:
                env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"})
            else:
                env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"})

            with open(name, 'w') as f[i]:
                pids[i] = subprocess.Popen(command.split(), env=env, stdout=f[i])
            if i == n_combinations - 1:
                last_process = True
            if ((i + 1) % n_parallel_threads == 0 and i >= n_parallel_threads - 1) or last_process:
                if last_process and not ((i + 1) % n_parallel_threads) == 0:
                    n_parallel_threads = (i + 1) % n_parallel_threads
                start = datetime.now()
                print('########## Waiting #############')
                for t in range(n_parallel_threads - 1, -1, -1):
                    pids[i - t].wait()
                end = datetime.now()
                print('########## Waiting Over######### Took', diff(end, start), 'for', n_parallel_threads, 'threads')

if __name__ == "__main__":
     run_algo()
