from os import path, makedirs, mkdir, listdir
import numpy as np
import collections
from datetime import datetime
import sys
import itertools

get_results_only = False
switch_gpus = False
n_parallel_threads = 1
args = dict()
args['hyper_params'] = ['DATA_DIR', 'ETA', 'ALPHA', 'BETA', 'THETA', 'GAMMA', 'LAMBDA', 'K', 'MAX_ITER']
custom = '_MNF_'
now = datetime.now()
args['timestamp'] = str(now.month)+'|'+str(now.day)+'|'+str(now.hour)+':'+str(now.minute)+':'+str(now.second) + custom

args['DATA_DIR'] = ['wiki']
args['ETA'] = [1.0]
args['ALPHA'] = [1.0]
args['BETA'] = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 10.0]
args['THETA'] = [1.0]
args['GAMMA'] = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 7.0, 10.0]
args['LAMBDA'] = [1.0]
args['K'] = [7]
args['MAX_ITER'] = [500]


if __name__ == "__main__":
    dir_name = ""
    global args
    global custom
    global now
    pos = args['hyper_params'].index('DATA_DIR')
    args['hyper_params'][0], args['hyper_params'][pos] = args['hyper_params'][pos], args['hyper_params'][0]

    stdout_dump_path = 'emb'
    param_values = []
    this_module = sys.modules[__name__]
    for hp_name in args['hyper_params']:
        param_values.append(args[hp_name])
    combinations = list(itertools.product(*param_values))
    n_combinations = len(combinations)
    print('Total no of experiments: ', n_combinations)
    all_result_files_lr = list()
    all_result_files_svm = list()
    all_result_files_n = list()
    merged_lr = collections.Counter(dict())
    tmp_lr_50 = collections.Counter(dict())
    merged_svm = collections.Counter(dict())
    tmp_svm_50 = collections.Counter(dict())
    merged_n = collections.Counter(dict())
    tmp_n_50 = collections.Counter(dict())

    config_n_50 = collections.Counter(dict())
    config_svm_50 = collections.Counter(dict())
    config_lr_50 = collections.Counter(dict())
    for i, setting in enumerate(combinations):
        folder_suffix = "MNF"
        dataset_name = ''
        for name, value in zip(args['hyper_params'], setting):
            folder_suffix += "_" + str(value)
            if name == 'DATA_DIR':
                dataset_name = value
        fldr = path.join(stdout_dump_path, dataset_name.title(), folder_suffix)
        sd1 = fldr
        sd2 = sd1 + "/Avg"
        file_names = [f for f in listdir(path.join(dir_name, sd2)) if path.isfile(path.join(dir_name, sd2, f))]
        no_of_files = len(file_names)

        for f in file_names:
            if f == "results_avg_lr.npy":
                result_file = np.load(path.join(dir_name, sd2, f)).item()
                result = result_file[folder_suffix][0][50]
                all_result_files_lr.append(result_file)
                tmp_lr_50[folder_suffix] = result['micro_f1']
                config_lr_50[folder_suffix] = result_file[folder_suffix][1]['0']

            elif f == "results_avg_svm.npy":
                result_file = np.load(path.join(dir_name, sd2, f)).item()
                result = result_file[folder_suffix][0][50]
                all_result_files_svm.append(result_file)
                tmp_svm_50[folder_suffix] = result['micro_f1']
                config_svm_50[folder_suffix] = result_file[folder_suffix][1]['0']


            elif f == "results_avg_n.npy":
                result_file = np.load(path.join(dir_name, sd2, f)).item()
                result = result_file[folder_suffix][0][50]
                all_result_files_n.append(result_file)
                tmp_n_50[folder_suffix] = result['micro_f1']
                config_n_50[folder_suffix] = result_file[folder_suffix][1]['0']

    tmp_n_50.most_common()

    tmp_lr_50.most_common()

    tmp_svm_50.most_common()


    for k, v in tmp_n_50.most_common(1):
        print('N :- %s: %f' % (k, v))
        print('Beta :- %s: Gamma :- %s K:- %s' % (config_n_50[k].BETA, config_n_50[k].GAMMA, config_n_50[k].K))
        print('Alpha :- %s: Theta :- %s K:- %s' % (config_n_50[k].ALPHA, config_n_50[k].THETA, config_n_50[k].K))
        if k not in merged_n:
            merged_n[k] = []

        merged_n[k].append(v)
    print()

    for k, v in tmp_lr_50.most_common(1):
        print('LR :- %s: %f' % (k, v))
        print('Beta :- %s: Gamma :- %s K:- %s' % (config_lr_50[k].BETA, config_lr_50[k].GAMMA, config_lr_50[k].K))
        print('Alpha :- %s: Theta :- %s K:- %s' % (config_lr_50[k].ALPHA, config_lr_50[k].THETA, config_lr_50[k].K))
        if k not in merged_lr:
            merged_lr[k] = []
        merged_lr[k].append(v)
    print()

    for k, v in tmp_svm_50.most_common(1):
        print('SVM :- %s: %f' % (k, v))
        print('Beta :- %s: Gamma :- %s K:- %s' % (config_svm_50[k].BETA, config_svm_50[k].GAMMA, config_svm_50[k].K))
        print('Alpha :- %s: Theta :- %s K:- %s' % (config_svm_50[k].ALPHA, config_svm_50[k].THETA, config_svm_50[k].K))
        if k not in merged_svm:
            merged_svm[k] = []
        merged_svm[k].append(v)
    print()

