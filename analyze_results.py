from os import path, makedirs, listdir
import numpy as np
import collections

if __name__ == "__main__":
    dir_name = ""
    sub_dir_name = ['wiki_n2v'] #'washington', 'wisconsin', 'texas', 'cornell', 'cora', 'citeseer', 'wiki', 'ppi', 'wiki_n2v']#, 'blogcatalog', 'armherst', 'hamilton', 'mich', 'rochester']
    extension = ["_C"]#, "_N"]
    for sd in sub_dir_name :
        for ex in extension :
            all_result_file = list()
            sd1 = sd + ex
            sd2 = sd1 + "/Avg"
            file_names = [f for f in listdir(path.join(dir_name, sd2)) if path.isfile(path.join(dir_name, sd2, f))]
            no_of_files = len(file_names)
            merged = collections.Counter(dict())
            tmp_10 = collections.Counter(dict())
            tmp_30 = collections.Counter(dict())
            tmp_50 = collections.Counter(dict())
            tmp_70 = collections.Counter(dict())
            for f in file_names:
                result_file = np.load(path.join(dir_name, sd2, f)).item()
                all_result_file.append(result_file)
                for k, v in result_file.items():  # Folder_suffix : accuracy
                    tmp_10[k] = v[0][10]['micro_precision']
                    tmp_30[k] = v[1][30]['micro_precision']
                    tmp_50[k] = v[2][50]['micro_precision']
                    tmp_70[k] = v[3][70]['micro_precision']
            tmp_10.most_common()
            tmp_30.most_common()
            tmp_50.most_common()
            tmp_70.most_common()
            for k, v in tmp_10.most_common(3):
                print('%s: %f' % (k, v))
                if k not in merged:
                    merged[k] = []
                merged[k].append(v)
            print()
            for k, v in tmp_30.most_common(3):
                print('%s: %f' % (k, v))
                if k not in merged:
                    merged[k] = []
                merged[k].append(v)
            print()
            for k, v in tmp_50.most_common(3):
                print('%s: %f' % (k, v))
                if k not in merged:
                    merged[k] = []
                merged[k].append(v)
            print()
            for k, v in tmp_70.most_common(3):
                print('%s: %f' % (k, v))
                if k not in merged:
                    merged[k] = []
                merged[k].append(v)
            # print(len(all_result_file))
            # print(merged)
            final_list = list()
            for k, v in merged.items():
                tuple_list = list()
                if len(v) == 3:
                    print("Three : ")
                    config = {}
                    for l in all_result_file:
                        if k in l.keys():
                            # if l.has_key(k):
                            config = l[k][4][str(0)]
                    #print("Improved over all three percentage of data --> Key : ", k, "Accuracy : ", v, "Config : ",
                          #config)
                    t1 = ("ALPHA", config.ALPHA)
                    t2 = ("BETA", config.BETA)
                    tuple_list.append(t1)
                    tuple_list.append(t2)
                    print(tuple_list)
                    final_list.append(tuple_list)
            for k, v in merged.items():
                tuple_list = list()
                if len(v) == 2:
                    print("Two : ")
                    config = {}
                    for l in all_result_file:
                        if k in l.keys():
                            # if l.has_key(k):
                            config = l[k][4][str(0)]
                    #print("Improved over upto two percentage of data --> Key : ", k, "Accuracy : ", v, "Config : ",
                          #config)
                    t1 = ("ALPHA", config.ALPHA)
                    t2 = ("BETA", config.BETA)
                    tuple_list.append(t1)
                    tuple_list.append(t2)
                    print(tuple_list)
                    final_list.append(tuple_list)
            for k, v in merged.items():
                tuple_list = list()
                if len(v) == 1:
                    print("One : ")
                    config = {}
                    for l in all_result_file:
                        if k in l.keys():
                            config = l[k][4][str(0)]
                    #print("Improved over upto one percentage of data --> Key : ", k, "Accuracy : ", v, "Config : ",
                          #config)
                    t1 = ("ALPHA", config.ALPHA)
                    t2 = ("BETA", config.BETA)
                    tuple_list.append(t1)
                    tuple_list.append(t2)
                    print(tuple_list)
                    final_list.append(tuple_list)
            print(final_list)
            print()
