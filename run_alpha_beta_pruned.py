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
n_parallel_threads = 16

# Set Hyper-parameters
args = dict()
master_args = dict()
# The names should be the same as argument names in parser.py
master_args['hyper_params'] = ['DATA_DIR', 'ETA', 'ALPHA', 'BETA', 'THETA', 'PHI', 'LAMBDA', 'L_COMPONENTS', 'K', 'COST_F']
args['hyper_params'] = ['DATA_DIR', 'ETA', 'THETA', 'PHI', 'LAMBDA', 'L_COMPONENTS', 'K', 'COST_F']
custom = '_LNMF_'
now = datetime.now()
args['timestamp'] = str(now.month)+'|'+str(now.day)+'|'+str(now.hour)+':'+str(now.minute)+':'+str(now.second) + custom #  '05|12|03:41:02'  # Month | Day | hours | minutes (24 hour clock)

args['DATA_DIR'] = ['ppi']# ,'washington', 'wisconsin', 'texas', 'cornell', 'ppi', 'wiki_n2v', 'cora', 'citeseer', 'wiki', 'armherst', 'hamilton', 'mich', 'rochester', 'blogcatalog']
args['ETA'] = [1.0]#, 1.0, 5.0]
#args['ALPHA'] = [0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0]
#args['BETA'] = [0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 10.0]
args['THETA'] = [1.0, 2.0, 5.0, 7.0, 10.0]
args['PHI'] = [1.0]#, 2.0, 5.0, 7.0, 10.0]
args['LAMBDA'] = [1.0] #[0.001, 0.01, 0.1, 1.0]
args['COST_F'] = [5e-5]#, 5e-4, 5e-3]
args['L_COMPONENTS'] = [128]#, 175]
args['K'] = [11]#, 10, 16, 24]

pos = args['hyper_params'].index('DATA_DIR')
args['hyper_params'][0], args['hyper_params'][pos] = args['hyper_params'][pos], args['hyper_params'][0]
#print(args)
cora_pruned_hyper_params = list()
cora_pruned_hyper_params.append([('ALPHA', '10.0'), ('BETA', '2.0')])
cora_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '0.5')])
cora_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '2.0')])
cora_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '0.1')])
cora_pruned_hyper_params.append([('ALPHA', '0.1'), ('BETA', '5.0')])
cora_pruned_hyper_params.append([('ALPHA', '5.0'), ('BETA', '7.0')])
cora_pruned_hyper_params.append([('ALPHA', '10.0'), ('BETA', '10.0')])
cora_pruned_hyper_params.append([('ALPHA', '1.0'), ('BETA', '5.0')])
cora_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '10.0')])
cora_pruned_hyper_params.append([('ALPHA', '0.1'), ('BETA', '1.0')])
cora_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '5.0')])
#pruned_hyper_params = cora_pruned_hyper_params

citeseer_pruned_hyper_params = list()
citeseer_pruned_hyper_params.append([('ALPHA', '1.0'), ('BETA', '10.0')])
citeseer_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '0.5')])
citeseer_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '5.0')])
citeseer_pruned_hyper_params.append([('ALPHA', '0.1'), ('BETA', '1.0')])
citeseer_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '0.1')])
citeseer_pruned_hyper_params.append([('ALPHA', '1.0'), ('BETA', '0.5')])
citeseer_pruned_hyper_params.append([('ALPHA', '5.0'), ('BETA', '0.1')])
citeseer_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '7.0')])
citeseer_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '7.0')])
#pruned_hyper_params = citeseer_pruned_hyper_params

wiki_pruned_hyper_params = list()
wiki_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '0.1')])
wiki_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '1.0')])
wiki_pruned_hyper_params.append([('ALPHA', '5.0'), ('BETA', '10.0')])
wiki_pruned_hyper_params.append([('ALPHA', '10.0'), ('BETA', '7.0')])
wiki_pruned_hyper_params.append([('ALPHA', '10.0'), ('BETA', '0.5')])
wiki_pruned_hyper_params.append([('ALPHA', '0.1'), ('BETA', '5.0')])
wiki_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '2.0')])
wiki_pruned_hyper_params.append([('ALPHA', '7.0'), ('BETA', '10.0')])
wiki_pruned_hyper_params.append([('ALPHA', '1.0'), ('BETA', '10.0')])
wiki_pruned_hyper_params.append([('ALPHA', '5.0'), ('BETA', '1.0')])


wiki_n2v_pruned_hyper_params = list()
wiki_n2v_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '7.0')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '2.0')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '1.0'), ('BETA', '0.5')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '0.5')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '1.0')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '1.0'), ('BETA', '10.0')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '7.0'), ('BETA', '0.5')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '1.0'), ('BETA', '1.0')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '0.5')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '10.0')])
wiki_n2v_pruned_hyper_params.append([('ALPHA', '1.0'), ('BETA', '0.1')])


ppi_pruned_hyper_params = list()
ppi_pruned_hyper_params.append([('ALPHA', '7.0'), ('BETA', '2.0')])
ppi_pruned_hyper_params.append([('ALPHA', '0.1'), ('BETA', '1.0')])
ppi_pruned_hyper_params.append([('ALPHA', '0.1'), ('BETA', '0.5')])
ppi_pruned_hyper_params.append([('ALPHA', '10.0'), ('BETA', '0.1')])
ppi_pruned_hyper_params.append([('ALPHA', '2.0'), ('BETA', '0.5')])
ppi_pruned_hyper_params.append([('ALPHA', '7.0'), ('BETA', '0.1')])
ppi_pruned_hyper_params.append([('ALPHA', '10.0'), ('BETA', '7.0')])
ppi_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '5.0')])
ppi_pruned_hyper_params.append([('ALPHA', '7.0'), ('BETA', '10.0')])
ppi_pruned_hyper_params.append([('ALPHA', '0.5'), ('BETA', '1.0')])
ppi_pruned_hyper_params.append([('ALPHA', '0.1'), ('BETA', '0.1')])
pruned_hyper_params = ppi_pruned_hyper_params


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
    combinations = list(itertools.product(*param_values))# list(tuples) ('washington', 1.0, 1.0, 1.0, 1.0, 128, 11, 5e-05)
    n_combinations = len(combinations) * len(pruned_hyper_params)
    print('Total no of experiments: ', n_combinations )

    pids = [None] * n_combinations
    f = [None] * n_combinations
    last_process = False

    # for itm in pruned_hyper_params:
    #     #print(itm) # [('ALPHA', 2.0), ('BETA', 5.0)]
    #     for it in itm:
    #         #print(it) # ('ALPHA', 2.0)
    #         command += "--" + it[0] + " " + str(it[1]) + " "
    #         folder_suffix += "_" + str(it[1])
    final_combinations = list()
    for itm in pruned_hyper_params:
        for it in combinations:
            final_combinations.append((it[0], it[1], itm[0][1], itm[1][1], it[2], it[3], it[4], it[5], it[6], it[7]))
    for i, setting in enumerate(final_combinations):
        # print(i, setting)
        fldr = ""
        command = "python main_algo.py "
        folder_suffix = "L"  # args['timestamp']
        for name, value in zip(master_args['hyper_params'], setting):
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
        # tabulate_results.write_results(args)

else:
    #tabulate_results.write_results(args)
    print("Done tabulation")

# Total no of experiments:  882
# 0 ('washington', 1.0, 1.0, 1.0, 1.0, 128, 11, 5e-05)
# 1 ('washington', 1.0, 1.0, 0.5, 1.0, 128, 11, 5e-05)
# 2 ('washington', 1.0, 1.0, 1.0, 1.0, 128, 11, 5e-05)
# 3 ('washington', 1.0, 1.0, 2.0, 1.0, 128, 11, 5e-05)
# 4 ('washington', 1.0, 1.0, 5.0, 1.0, 128, 11, 5e-05)
# 5 ('washington', 1.0, 1.0, 7.0, 1.0, 128, 11, 5e-05)
# 6 ('washington', 1.0, 1.0, 10.0, 1.0, 128, 11, 5e-05)
# 7 ('washington', 1.0, 0.5, 1.0, 1.0, 128, 11, 5e-05)
# 8 ('washington', 1.0, 0.5, 0.5, 1.0, 128, 11, 5e-05)
# 9 ('washington', 1.0, 0.5, 1.0, 1.0, 128, 11, 5e-05)
# 10 ('washington', 1.0, 0.5, 2.0, 1.0, 128, 11, 5e-05)
# 11 ('washington', 1.0, 0.5, 5.0, 1.0, 128, 11, 5e-05)
# 12 ('washington', 1.0, 0.5, 7.0, 1.0, 128, 11, 5e-05)
# 13 ('washington', 1.0, 0.5, 10.0, 1.0, 128, 11, 5e-05)
# 14 ('washington', 1.0, 1.0, 1.0, 1.0, 128, 11, 5e-05)
# 15 ('washington', 1.0, 1.0, 0.5, 1.0, 128, 11, 5e-05)
# 16 ('washington', 1.0, 1.0, 1.0, 1.0, 128, 11, 5e-05)
# 17 ('washington', 1.0, 1.0, 2.0, 1.0, 128, 11, 5e-05)
# 18 ('washington', 1.0, 1.0, 5.0, 1.0, 128, 11, 5e-05)
# 19 ('washington', 1.0, 1.0, 7.0, 1.0, 128, 11, 5e-05)
# 20 ('washington', 1.0, 1.0, 10.0, 1.0, 128, 11, 5e-05)
# 21 ('washington', 1.0, 2.0, 1.0, 1.0, 128, 11, 5e-05)
# 22 ('washington', 1.0, 2.0, 0.5, 1.0, 128, 11, 5e-05)
# 23 ('washington', 1.0, 2.0, 1.0, 1.0, 128, 11, 5e-05)
# 24 ('washington', 1.0, 2.0, 2.0, 1.0, 128, 11, 5e-05)
# 25 ('washington', 1.0, 2.0, 5.0, 1.0, 128, 11, 5e-05)
# 26 ('washington', 1.0, 2.0, 7.0, 1.0, 128, 11, 5e-05)
# 27 ('washington', 1.0, 2.0, 10.0, 1.0, 128, 11, 5e-05)
# 28 ('washington', 1.0, 5.0, 1.0, 1.0, 128, 11, 5e-05)
# 29 ('washington', 1.0, 5.0, 0.5, 1.0, 128, 11, 5e-05)
# 30 ('washington', 1.0, 5.0, 1.0, 1.0, 128, 11, 5e-05)
# 31 ('washington', 1.0, 5.0, 2.0, 1.0, 128, 11, 5e-05)
# 32 ('washington', 1.0, 5.0, 5.0, 1.0, 128, 11, 5e-05)
# 33 ('washington', 1.0, 5.0, 7.0, 1.0, 128, 11, 5e-05)
# 34 ('washington', 1.0, 5.0, 10.0, 1.0, 128, 11, 5e-05)
# 35 ('washington', 1.0, 7.0, 1.0, 1.0, 128, 11, 5e-05)
# 36 ('washington', 1.0, 7.0, 0.5, 1.0, 128, 11, 5e-05)
# 37 ('washington', 1.0, 7.0, 1.0, 1.0, 128, 11, 5e-05)
# 38 ('washington', 1.0, 7.0, 2.0, 1.0, 128, 11, 5e-05)
# 39 ('washington', 1.0, 7.0, 5.0, 1.0, 128, 11, 5e-05)
# 40 ('washington', 1.0, 7.0, 7.0, 1.0, 128, 11, 5e-05)
# 41 ('washington', 1.0, 7.0, 10.0, 1.0, 128, 11, 5e-05)
# 42 ('washington', 1.0, 10.0, 1.0, 1.0, 128, 11, 5e-05)
# 43 ('washington', 1.0, 10.0, 0.5, 1.0, 128, 11, 5e-05)
# 44 ('washington', 1.0, 10.0, 1.0, 1.0, 128, 11, 5e-05)
# 45 ('washington', 1.0, 10.0, 2.0, 1.0, 128, 11, 5e-05)
# 46 ('washington', 1.0, 10.0, 5.0, 1.0, 128, 11, 5e-05)
# 47 ('washington', 1.0, 10.0, 7.0, 1.0, 128, 11, 5e-05)
# 48 ('washington', 1.0, 10.0, 10.0, 1.0, 128, 11, 5e-05)


