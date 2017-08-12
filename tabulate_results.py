import xlwt
import itertools
import numpy as np
from os import path, mkdir, listdir

save_path = 'resuts_xls'
if not path.exists(save_path):
    mkdir(save_path)


# Need to pass the args dictionary
#   - run.py passes it or - can load it from args directory
#   - if none, loads it from a specified folder
def write_results(fldr, args, folder=None):

    book = xlwt.Workbook(encoding='utf-8')

    sheets = {}
    metric_names = ['accuracy', 'micro_f1', 'macro_f1', 'cross_entropy', 'hamming_loss', 'micro_precision', 'micro_recall',
            'macro_precision', 'macro_recall',  'coverage', 'pak', 'ranking_loss',
            'average_precision', 'bae']
    n_metrics = len(metric_names)
    cols = ['', 'labels', 'percents', 'folds', ''] + metric_names

    if args is not None:
        cols = args['hyper_params'][1:] + cols

        param_values = []
        for hp_name in args['hyper_params'][1:]:
            param_values.append(args[hp_name])
        combinations = list(itertools.product(*param_values))

        for data_name in args['DATA_DIR']:
            sheets[data_name] = book.add_sheet(data_name, cell_overwrite_ok=True)
            sheets[data_name+'_c'] = book.add_sheet(data_name+'_c', cell_overwrite_ok=True)
            sheets[data_name + '_avg'] = book.add_sheet(data_name + '_avg', cell_overwrite_ok=True)
            sheets[data_name + '_c_avg'] = book.add_sheet(data_name + '_c_avg', cell_overwrite_ok=True)

            # Write Header names
            row0 = sheets[data_name].row(0)
            row_c_0 = sheets[data_name+'_c'].row(0)
            row_a0 = sheets[data_name+'_avg'].row(0)
            row_a_c_0 = sheets[data_name + '_c_avg'].row(0)
            col_id = -1
            for header in cols:
                col_id += 1
                row0.write(col_id, header)
                row_a0.write(col_id, header)
                row_c_0.write(col_id, header)
                row_a_c_0.write(col_id, header)

            row_id = 0
            row_a_id = 0
            for setting in combinations:
                folder_suffix = ''
                row = sheets[data_name].row(row_id + 1)
                row_a = sheets[data_name+'_avg'].row(row_a_id + 1)
                row_c_0 = sheets[data_name+'_c'].row(row_id + 1)
                row_a_c_0 = sheets[data_name + '_c_avg'].row(row_a_id + 1)


                for name, value in zip(args['hyper_params'][1:], setting):
                    folder_suffix += "_" + str(value)
                #   print(path.join(data_name, args['timestamp'] + folder_suffix))
                if not path.exists(path.join(data_name, args['timestamp'] + folder_suffix)):
                    #row_a_id += 1
                    continue


                for name, value in zip(args['hyper_params'][1:], setting):
                    row.write(cols.index(name), value)
                    row_a.write(cols.index(name), value)
                    row_c_0.write(cols.index(name), value)
                    row_a_c_0.write(cols.index(name), value)

                folder_suffix = path.join(data_name, args['timestamp']+folder_suffix)

               
                results = np.load(path.join(fldr, 'results.npy')).item()
                results_c = np.load(path.join(fldr, 'results_c.npy')).item()
               

                row = sheets[data_name].row(row_id + 1)
                row_a = sheets[data_name + '_avg'].row(row_a_id + 1)
                row_c_0 = sheets[data_name + '_c'].row(row_id + 1)
                row_a_c_0 = sheets[data_name + '_c_avg'].row(row_a_id + 1)
                row.write(cols.index('labels'), "Random")
                row_a.write(cols.index('labels'), "Random")
                row_c_0.write(cols.index('labels'), "Random")
                row_a_c_0.write(cols.index('labels'), "Random")

                percents = np.sort([int(key) for key in results.keys()]).astype(np.str_)
                folds = np.sort([int(fold) for fold in results[str(percents[0])].keys()]).astype(np.str_)

                perc_pos = cols.index('percents')
                fold_pos = cols.index('folds')

                for pid, percent in enumerate(percents):
                    row = sheets[data_name].row(row_id + 1)
                    row.write(perc_pos, int(percent))
                    row_a = sheets[data_name + '_avg'].row(row_a_id + 1)
                    row_a.write(perc_pos, int(percent))
                    row_c_0 = sheets[data_name + '_c'].row(row_id + 1)
                    row_c_0.write(perc_pos, int(percent))
                    row_a_c_0 = sheets[data_name + '_c_avg'].row(row_a_id + 1)
                    row_a_c_0.write(perc_pos, int(percent))

                    mean_metrics = np.zeros([1, (pid + 1) * n_metrics])
                    mean_metrics_c = np.zeros([1, (pid + 1) * n_metrics])
                    for fold in folds:
                        row_id += 1
                        row = sheets[data_name].row(row_id)
                        row.write(fold_pos, int(fold))
                        row_c_0 = sheets[data_name + '_c'].row(row_id)
                        row_c_0.write(fold_pos, int(fold))

                        offset = pid * len(cols)
                        order = []
                        for metric in metric_names:
                            order.append(cols.index(metric) - (5 + len(args['hyper_params'][1:])))
                            val = float(results[percent][fold][metric])
                            val_c = float(results_c[percent][fold][metric])

                            row.write(cols.index(metric), round(val, 5))
                            row_c_0.write(cols.index(metric), round(val_c, 5))
                            mean_metrics[0, offset + cols.index(metric) - (5 + len(args['hyper_params'][1:]))] += round(
                                val, 5)
                            mean_metrics_c[0, offset + cols.index(metric) - (5 + len(args['hyper_params'][1:]))] += round(
                                val_c, 5)

                    row_id += 1
                    row_a_id += 1
                    row = sheets[data_name].row(row_id)
                    row_a = sheets[data_name + '_avg'].row(row_a_id)
                    row_c_0 = sheets[data_name + '_c'].row(row_id)
                    row_a_c_0 = sheets[data_name + '_c_avg'].row(row_a_id)

                    for i, metric in enumerate(metric_names):
                        val = mean_metrics[0, order[i]]
                        val_c = mean_metrics_c[0, order[i]]
                        row.write(cols.index(metric), val / len(folds))
                        row_a.write(cols.index(metric), val / len(folds))
                        row_c_0.write(cols.index(metric), val_c / len(folds))
                        row_a_c_0.write(cols.index(metric), val_c / len(folds))

                    row_id += 1
                    row = sheets[data_name].row(row_id)
                    row_c_0 = sheets[data_name + '_c'].row(row_id)
                    for i in range(len(cols)):
                        row.write(i, '')
                        row_c_0.write(i, '')


    # Save it with a time-stamp
    book.save(path.join(save_path, args['timestamp']+'.xls'))
    #book.save(path.join(save_path, 'default.xls'))

