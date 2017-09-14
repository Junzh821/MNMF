from os import path, makedirs, mkdir, listdir
import numpy as np
import collections
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sys
import itertools

get_results_only = False
switch_gpus = False #For multiple GPUs
n_parallel_threads = 4
# Set Hyper-parameters
args = dict()
# The names should be the same as argument names in parser.py
args['hyper_params'] = ['DATA_DIR', 'ETA', 'ALPHA', 'BETA', 'THETA', 'GAMMA', 'LAMBDA', 'K', 'CONV_LS', 'MAX_ITER']
custom = '_LNMF_'
now = datetime.now()
args['timestamp'] = str(now.month)+'|'+str(now.day)+'|'+str(now.hour)+':'+str(now.minute)+':'+str(now.second) + custom #  '05|12|03:41:02'  # Month | Day | hours | minutes (24 hour clock)

args['DATA_DIR'] = ['cora'] #'washington', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'wiki', 'ppi', 'wiki_n2v', 'blogcatalog', 'armherst', 'hamilton', 'mich', 'rochester']
args['ETA'] = [1.0]
args['ALPHA'] = [0.1, 0.5, 1.0, 2.0, 5.0]
args['BETA'] = [0.1, 0.5, 1.0, 2.0, 5.0]
args['THETA'] = [0.1, 0.5, 1.0, 2.0, 5.0]
args['GAMMA'] = [1.0]#0.1, 0.5, 1.0, 2.0, 5.0]
args['LAMBDA'] = [1.0]#0.01, 0.1, 1.0]
args['K'] = [7]#5, 7, 10, 15, 20]
args['CONV_LS'] = [7]#5, 7, 10, 15, 20]
args['MAX_ITER'] = [11]#, 25, 40, 75]

if __name__ == "__main__":
    dir_name = ""
    global args
    global custom
    global now
    pos = args['hyper_params'].index('DATA_DIR')
    args['hyper_params'][0], args['hyper_params'][pos] = args['hyper_params'][pos], args['hyper_params'][0]

    # Create Log Directory for stdout Dumps
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
    tmp_lr_10 = collections.Counter(dict())
    tmp_lr_30 = collections.Counter(dict())
    tmp_lr_50 = collections.Counter(dict())
    tmp_lr_70 = collections.Counter(dict())
    merged_svm = collections.Counter(dict())
    tmp_svm_10 = collections.Counter(dict())
    tmp_svm_30 = collections.Counter(dict())
    tmp_svm_50 = collections.Counter(dict())
    tmp_svm_70 = collections.Counter(dict())
    merged_n = collections.Counter(dict())
    tmp_n_10 = collections.Counter(dict())
    tmp_n_30 = collections.Counter(dict())
    tmp_n_50 = collections.Counter(dict())
    tmp_n_70 = collections.Counter(dict())

    for i, setting in enumerate(combinations):
        folder_suffix = "MNF-L"
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
                all_result_files_lr.append(result_file)
                for k, v in result_file.items():  # Folder_suffix : accuracy
                    # print(v[0][10]['micro_f1'])
                    tmp_lr_10[k] = v[0][10]['micro_f1']
                    tmp_lr_30[k] = v[1][30]['micro_f1']
                    tmp_lr_50[k] = v[2][50]['micro_f1']
                    tmp_lr_70[k] = v[3][70]['micro_f1']
            elif f == "results_avg_svm.npy":
                result_file = np.load(path.join(dir_name, sd2, f)).item()
                all_result_files_svm.append(result_file)
                for k, v in result_file.items():  # Folder_suffix : accuracy
                    # print(v[0][10]['micro_f1'])
                    tmp_svm_10[k] = v[0][10]['micro_f1']
                    tmp_svm_30[k] = v[1][30]['micro_f1']
                    tmp_svm_50[k] = v[2][50]['micro_f1']
                    tmp_svm_70[k] = v[3][70]['micro_f1']
            elif f == "results_avg_n.npy":
                result_file = np.load(path.join(dir_name, sd2, f)).item()
                all_result_files_n.append(result_file)
                for k, v in result_file.items():  # Folder_suffix : accuracy
                    # print(v[0][10]['micro_f1'])
                    tmp_n_10[k] = v[0][10]['micro_f1']
                    tmp_n_30[k] = v[1][30]['micro_f1']
                    tmp_n_50[k] = v[2][50]['micro_f1']
                    tmp_n_70[k] = v[3][70]['micro_f1']
    tmp_n_10.most_common()
    tmp_n_30.most_common()
    tmp_n_50.most_common()
    tmp_n_70.most_common()
    tmp_lr_10.most_common()
    tmp_lr_30.most_common()
    tmp_lr_50.most_common()
    tmp_lr_70.most_common()
    tmp_svm_10.most_common()
    tmp_svm_30.most_common()
    tmp_svm_50.most_common()
    tmp_svm_70.most_common()

    for k, v in tmp_n_10.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_n:
            merged_n[k] = []
        merged_n[k].append(v)
    print()
    for k, v in tmp_n_30.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_n:
            merged_n[k] = []
        merged_n[k].append(v)
    print()
    for k, v in tmp_n_50.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_n:
            merged_n[k] = []
        merged_n[k].append(v)
    print()
    for k, v in tmp_n_70.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_n:
            merged_n[k] = []
        merged_n[k].append(v)

    for k, v in tmp_lr_10.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_lr:
            merged_lr[k] = []
        merged_lr[k].append(v)
    print()
    for k, v in tmp_lr_30.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_lr:
            merged_lr[k] = []
        merged_lr[k].append(v)
    print()
    for k, v in tmp_lr_50.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_lr:
            merged_lr[k] = []
        merged_lr[k].append(v)
    print()
    for k, v in tmp_lr_70.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_lr:
            merged_lr[k] = []
        merged_lr[k].append(v)

    for k, v in tmp_svm_10.most_common(1):
        print('\n\n%s: %f' % (k, v))
        if k not in merged_svm:
            merged_svm[k] = []
        merged_svm[k].append(v)
    print()
    for k, v in tmp_svm_30.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_svm:
            merged_svm[k] = []
        merged_svm[k].append(v)
    print()
    for k, v in tmp_svm_50.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_svm:
            merged_svm[k] = []
        merged_svm[k].append(v)
    print()
    for k, v in tmp_svm_70.most_common(1):
        print('%s: %f' % (k, v))
        if k not in merged_svm:
            merged_svm[k] = []
        merged_svm[k].append(v)