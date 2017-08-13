import argparse
import logging
import time
import sys
from os import path, makedirs, listdir, mkdir
from scipy.io import loadmat
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import data_utils as du
import eval_performance as ep
import xlsxwriter
import MNMF as mnmf
import latent_graphs as lg
import collections
import json
from sklearn.linear_model import LogisticRegression

def get_ArgumentParser():
    parser = argparse.ArgumentParser()

    # Experiment configuration
    parser.add_argument("--DATA_DIR", default="washington")
    # webkb-dataset-washington, citeseer, cora, wiki, armherst, ppi
    parser.add_argument("--LOG_DIR", default="Results/")
    parser.add_argument("--LOG_FNAME", default="mod_mnmf.log")

    # Factorization
    parser.add_argument("--LG", default=1, help="Latent graph integration ")
    parser.add_argument("--FACT_Y", default=1, help="Factorize Label matrix")
    parser.add_argument("--ENF_Y", default=0, help="Enforce Label matrix")

    # Weights for different components in the objective functions
    parser.add_argument("--ETA", default=5.0, help="First and Second order proximity mixing parameters")
    parser.add_argument("--ALPHA", default=2.0, help="Similarity matrix factorization weight")
    parser.add_argument("--DELTA", default=1.0, help="Attribute matrix factorization weight")
    parser.add_argument("--BETA", default=1.0, help="Community Indicator matrix factorization weight")
    parser.add_argument("--GAMMA", default=1.0, help="Modularity Maximization weight")
    parser.add_argument("--THETA", default=5.0, help="Label matrix factorization weight")
    parser.add_argument("--PHI", default=3.0, help="Network regularization weight")
    parser.add_argument("--LAMBDA", default=0.1, help="L2 regularization weight for S and H factorization")
    parser.add_argument("--ZETA", default=1e+9, help="L2 regularization weight for solving H")
    # paper to code attribute mapping : alpha for S factorization, beta for H factorization, gamma for modularity maximization

    # Factorization parameters
    parser.add_argument("--MAX_ITER", default=700)
    parser.add_argument("--L_COMPONENTS", default=150)
    parser.add_argument("--K", default=10)
    parser.add_argument("--INIT", default="nndsvd")
    parser.add_argument("--PROJ", default=True)
    parser.add_argument("--COST_F", default='LS')
    parser.add_argument("--CONV_LS", default=7e-5)
    parser.add_argument("--CONV_KL", default=1e-4)
    parser.add_argument("--CONV_MUL", default=1e-4)
    parser.add_argument("--MULTI_LABEL", default=False)
    #parser.add_argument("--HYPER_STR", default=" ")

    return parser

def init(dir_name, file_name):
    if not path.exists(dir_name):
        makedirs(dir_name)
    #file_name = path.join(dir_name, file_name)
    file_name = dir_name + "_" + file_name
    logging.basicConfig(filename=file_name, filemode='w', level=logging.DEBUG)
    return logging.getLogger('M-NMF')

def load_dataset(dir_name):
    #Get File name
    attribute_names = [f for f in listdir(path.join(dir_name, 'Attributes')) if path.isfile(path.join(dir_name, 'Attributes', f))]
    relation_names = [f for f in listdir(path.join(dir_name, 'Relation')) if path.isfile(path.join(dir_name, 'Relation', f))]
    expt_sets = np.array([f for f in listdir(path.join(dir_name, 'index')) if not path.isfile(path.join(dir_name, 'index', f))], dtype=np.int32)
    expt_sets = np.sort(expt_sets)
    n_folds = len([1 for f in listdir(path.join(dir_name, 'index', str(expt_sets[0]))) if not path.isfile(path.join(dir_name, 'index', f))])
    print("================ Dataset Details : Start ================")
    print("Dataset: %s" % dir_name.split("/")[2])
    print("Attributes: %s" % attribute_names)
    print("Relations: %s" % relation_names)
    print("Sets: %s" % expt_sets)
    print("N_Folds: %d" % n_folds)

    #Load data :  all are in (nxn), (nxq), (nxm) format
    attributes = [loadmat(path.join(dir_name, 'Attributes', name))['view'] for name in attribute_names]
    relations = [loadmat(path.join(dir_name, 'Relation', name))['adjmat'] for name in relation_names]
    for idx in range(len(relations)) :
        if issparse(relations[idx]) :
            relations[idx] = relations[idx].toarray().astype(np.float64)

    #relations - make them undirected graph
    #relations = [du.make_undirected(relation) for relation in relations]
    truth = loadmat(path.join(dir_name, 'truth'))['truth']
    n_ids, n_labels = np.shape(truth)
    n_features = [np.shape(attr)[1] for attr in attributes]
    print("Number of nodes : %d" % n_ids)
    print("Number of labels : %d" % n_labels)
    print("Number of features : %s" % n_features)
    all_label_dist = du.get_all_label_distribution(truth)
    print("All label distribution : %s" % all_label_dist)
    datasets_template = collections.namedtuple('Datasets_template', ['attributes', 'relations', 'truth', 'expt_sets', \
                                                                     'n_folds', 'n_ids', 'n_labels', 'n_features', \
                                                                     'n_views', 'n_relations', 'all_label_dist'])

    dataset = datasets_template(attributes=attributes, relations=relations, truth=truth, expt_sets=expt_sets, \
                                n_folds=n_folds, n_ids=n_ids, n_labels=n_labels, n_features=n_features, n_views=len(attributes), \
                                n_relations=len(relations), all_label_dist=all_label_dist)
    print("================ Dataset Details : End ================")
    return dataset

def get_perf_metrics_using_classifier(config, entity_embedding, labels, train_ids, val_ids, test_ids) :
    pred_ids = test_ids
    #pred_ids = np.logical_not(train_ids)
    labelled_ids = np.logical_or(val_ids, train_ids)
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(entity_embedding[labelled_ids, :], labels[labelled_ids, :])
    predictions = clf.predict_proba(entity_embedding[pred_ids, :])
    performances = ep.evaluate(predictions, labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return performances

def get_perf_metrics(config, entity_embedding, Q, labels, train_ids, val_ids, test_ids) :
    pred_ids = test_ids
    #pred_ids = np.logical_not(train_ids)
    labelled_ids = np.logical_or(val_ids, train_ids)
    Y_hat = np.dot(Q, entity_embedding.T)
    Y_hat = Y_hat.T
    #Y_hat_binarized = np.zeros_like(Y_hat)
    #Y_hat_binarized[np.arange(len(Y_hat)), Y_hat.argmax(1)] = 1
    perf = ep.evaluate(Y_hat[pred_ids, :], labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return perf

def tune_model_using_classifier(config, entity_embedding, labels, train_ids, val_ids) :
    pred_ids = val_ids
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(entity_embedding[train_ids, :], labels[train_ids, :])
    predictions = clf.predict_proba(entity_embedding[pred_ids, :])
    performances = ep.evaluate(predictions, labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return performances

def tune_model(config, entity_embedding,  Q, labels, train_ids, val_ids) :
    pred_ids = val_ids
    Y_hat = np.dot(Q, entity_embedding.T)
    Y_hat = Y_hat.T
    #Y_hat_binarized = np.zeros_like(Y_hat)
    #Y_hat_binarized[np.arange(len(Y_hat)), Y_hat.argmax(1)] = 1
    performances = ep.evaluate(Y_hat[pred_ids, :], labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return performances

# def plot_latent_weights(master_latent_weights, perc_data):
#     plt.axis([-10, 100, -1.0, 1.0  ])
#     plt.grid(True)
#     plt.title('latent_weights')
#     plt.ylabel('weights')
#     plt.xlabel('percentage of labeled data')
#     for method in master_latent_weights.keys():
#         plt.plot(perc_data, master_latent_weights[method][1], master_latent_weights[method][0])
#     plt.savefig('Results/plot_latent_weights.png')
#     plt.show()

def get_LatentGraph(lg_methods, dataset, labels, train_ids):
    n_ids = dataset.n_ids
    attributes = dataset.attributes
    relations = dataset.relations
    latent_graphs = []
    weights = []

    erw_alpha = 0.1  # ERWR Restart probability
    erw_iters = 10  # ERWR number of iterations
    comb_attribute = np.array([]).reshape(n_ids, 0)
    n_labelled = np.count_nonzero(train_ids)
    # if multiple attribute views are present, combine them by stacking
    for attribute in attributes:
        comb_attribute = np.hstack((comb_attribute, attribute))

    rel_ids = 1
    for relation in relations:
        tmp_lgs = []
        # Add the original graph too
        tmp_lgs.append(relation)
        lg_methods_params = {'ERW': (relation, erw_alpha, erw_iters), 'pred_sim': (n_ids, comb_attribute, labels, train_ids), \
         'att_sim': (comb_attribute,), 'label_sim': (n_labelled, relation, labels, train_ids)}
        lg_methods_params = collections.OrderedDict(sorted(lg_methods_params.items()))

        for method in lg_methods.keys():
            tmp_lgs.append(lg_methods[method](*lg_methods_params[method]))

        tmp, w = lg.integrate_latent_graphs(n_ids, tmp_lgs, labels, train_ids)
        weights.append(w)
        latent_graphs.append(np.copy(tmp))
        print("Relation: %d, Weights: %s" % (rel_ids, w))
        rel_ids += 1

    # latent_graph = lg.integrate_latent_graphs(n_ids, latent_graphs, labels, train_ids)
    return latent_graphs[0], weights[0]

def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Matrix Factorization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    config = get_ArgumentParser().parse_args()
    if config.DATA_DIR in ['washington', 'wisconsin', 'texas', 'cornell', 'armherst', 'rochester', 'mich', 'hamilton',
                           'citeseer', 'cora', 'wiki']:
        config.MULTI_LABEL = False
    elif config.DATA_DIR in ['ppi', 'blogcatalog', 'wiki_n2v']:
        config.MULTI_LABEL = True
    fldr1 = config.DATA_DIR + "_C"
    if not path.exists(fldr1):
        makedirs(fldr1, exist_ok=True)
    fldr2 = config.DATA_DIR + "_N"
    if not path.exists(fldr2):
        makedirs(fldr2, exist_ok=True)
    fldr3 = path.join(fldr1, "Avg")
    if not path.exists(fldr3):
        makedirs(fldr3, exist_ok=True)
    fldr4 = path.join(fldr2, "Avg")
    if not path.exists(fldr4):
        makedirs(fldr4, exist_ok=True)
    old_data_dir = config.DATA_DIR
    config.DATA_DIR = path.join("../Datasets/", config.DATA_DIR)
    print("Config: %s" % (config))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Initialization and loading dataset
    logger = init(config.LOG_DIR, config.LOG_FNAME)
    dataset = load_dataset(config.DATA_DIR)
    h_row = 2
    names_c = ["MNMF_c.xlsx"]
    tmp = {"accuracy": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0, "macro_precision": 0,
           "macro_recall": 0, "macro_f1": 0, "average_precision": 0, "coverage": 0, "ranking_loss": 0,
           "hamming_loss": 0, "cross_entropy": 0}
    overall_performances_c_0 = tmp
    overall_performances_c = [overall_performances_c_0]
    # workbooks = []
    # worksheets = []
    # for i in range(len(names)) :
    #     workbook = xlsxwriter.Workbook(path.join(config.LOG_DIR, names[i]))
    #     worksheet = workbook.add_worksheet()
    #     worksheets.append(worksheet)
    #     workbooks.append(workbook)
    # for i in range(len(names_c)) :
    #     workbook = xlsxwriter.Workbook(path.join(config.LOG_DIR, names_c[i]))
    #     worksheet = workbook.add_worksheet()
    #     worksheets.append(worksheet)
    #     workbooks.append(workbook)
    # metrics = ["Accuracy", "Micro Precision", "Micro Recall", "Micro F1", "Macro Precision", "Macro Recall", "Macro F1",
    #            "Average Precision", "Coverage", "Ranking Loss", "Hamming Loss", "Cross Entropy"]
    # for a in range(len(metrics)):
    #     for i in range(len(worksheets)):
    #         worksheets[i].write(0, a + 1, metrics[a])
    #         worksheets[i].set_column(0, a + 1, 20)
    perc_data = dataset.expt_sets
    all_results_c = {}
    all_avg_results_c = {}
    l_res_c = list()
    master_results = {}
    for a in perc_data :
        all_results_c[a] = {}
        all_avg_results_c[str.split(config.LOG_DIR, "/")[1]] = list()
        h_col = 0
        master_results['C'] = list()
        # for i in range(len(worksheets)):
        #     worksheets[i].write(h_row, h_col, a)
        overall_performances_c = [dict.fromkeys(aaa, 0) for aaa in overall_performances_c]
        itr = 0
        print("% of randomly sampled training data ---- ", a)
        for b in range(1, dataset.n_folds+1) :
            data_dir = path.join(config.DATA_DIR, 'index', str(a), str(b))
            train_ids = np.load(path.join(data_dir, 'train_ids.npy')).astype(dtype=bool)
            val_ids = np.load(path.join(data_dir, 'val_ids.npy')).astype(dtype=bool)
            test_ids = np.load(path.join(data_dir, 'test_ids.npy')).astype(dtype=bool)
            unlabelled_ids = np.logical_or(val_ids, test_ids)
            n_unlabelled = np.count_nonzero(unlabelled_ids)
            labels = np.copy(dataset.truth)
            labels[unlabelled_ids, :] = np.zeros((n_unlabelled, dataset.n_labels))

            Y = dataset.truth
            Y_train = labels
            D = [i.T for i in dataset.attributes]  # mxn
            X = [i for i in dataset.relations]  # nxn

            h_col = 1
            entity_embeddings = []
            performances = []
            best_result_c = mnmf.factorize(config, D, X, Y.T, Y_train.T, train_ids, val_ids, logger)
            entity_embeddings.append(best_result_c['U'])
            # outputEntities = path.join(config.LOG_DIR, "U_" + str(a) + "_" + str(b) + "_" + "_c.log")  # U
            # np.savetxt(outputEntities, best_result_c['U'], fmt="%f")
            # outputEntities = path.join(config.LOG_DIR, "H_" + str(a) + "_" + str(b) + "_" + "_c.log")  # U
            # np.savetxt(outputEntities, best_result_c['H'], fmt="%f")
            # outputEntities = path.join(config.LOG_DIR, "Q_" + str(a) + "_" + str(b) + "_" + "_c.log")  # U
            # np.savetxt(outputEntities, best_result_c['Q'], fmt="%f")
            # outputEntities = path.join(config.LOG_DIR, "U_" + str(a) + "_" + str(b) + "_" + "_.log")  # U
            # np.savetxt(outputEntities, best_result['U'], fmt="%f")
            # outputEntities = path.join(config.LOG_DIR, "H_" + str(a) + "_" + str(b) + "_" + "_.log")  # U
            # np.savetxt(outputEntities, best_result['H'], fmt="%f")
            # outputEntities = path.join(config.LOG_DIR, "Q_" + str(a) + "_" + str(b) + "_" + "_.log")  # U
            # np.savetxt(outputEntities, best_result['Q'], fmt="%f")

            for i in range(len(entity_embeddings)) :
                performance = get_perf_metrics_using_classifier(config, best_result_c['U'], Y, train_ids, val_ids, test_ids)
                print("Performance_using_classifier : Test accuracy: {%0.5f } , Test Loss: {%0.5f }" % (
                    performance['accuracy'], performance['cross_entropy']))
                performances.append(performance)
                all_results_c[a][b] = performance
            for i in range(len(overall_performances_c)) :
                if len(overall_performances_c) == len(performances) :
                    overall_performances_c[i]["accuracy"] += performances[i]["accuracy"]
                    overall_performances_c[i]["micro_precision"] += performances[i]["micro_precision"]
                    overall_performances_c[i]["micro_recall"] += performances[i]["micro_recall"]
                    overall_performances_c[i]["micro_f1"] += performances[i]["micro_f1"]
                    overall_performances_c[i]["macro_precision"] += performances[i]["macro_precision"]
                    overall_performances_c[i]["macro_recall"] += performances[i]["macro_recall"]
                    overall_performances_c[i]["macro_f1"] += performances[i]["macro_f1"]
                    overall_performances_c[i]["average_precision"] += performances[i]["average_precision"]
                    overall_performances_c[i]["coverage"] += performances[i]["coverage"]
                    overall_performances_c[i]["ranking_loss"] += performances[i]["ranking_loss"]
                    overall_performances_c[i]["hamming_loss"] += performances[i]["hamming_loss"]
                    overall_performances_c[i]["cross_entropy"] += performances[i]["cross_entropy"]
            # for i in range(1, len(worksheets)):
            #     worksheets[i].write(h_row, h_col, performances[i-1]["accuracy"])
            #     worksheets[i].write(h_row, h_col + 1, performances[i-1]["micro_precision"])
            #     worksheets[i].write(h_row, h_col + 2, performances[i-1]["micro_recall"])
            #     worksheets[i].write(h_row, h_col + 3, performances[i-1]["micro_f1"])
            #     worksheets[i].write(h_row, h_col + 4, performances[i-1]["macro_precision"])
            #     worksheets[i].write(h_row, h_col + 5, performances[i-1]["macro_recall"])
            #     worksheets[i].write(h_row, h_col + 6, performances[i-1]["macro_f1"])
            #     worksheets[i].write(h_row, h_col + 7, performances[i-1]["average_precision"])
            #     worksheets[i].write(h_row, h_col + 8, performances[i-1]["coverage"])
            #     worksheets[i].write(h_row, h_col + 9, performances[i-1]["ranking_loss"])
            #     worksheets[i].write(h_row, h_col + 10, performances[i-1]["hamming_loss"])
            #     worksheets[i].write(h_row, h_col + 11, performances[i-1]["cross_entropy"])
            print("**********************************************************")
            h_row += 1
            itr += 1
        overall_performances_c = [{k: v / dataset.n_folds for k, v in d.items()} for d in overall_performances_c]
        print(overall_performances_c)
        l_res_c.append({a: overall_performances_c[0]})
        # for i in range(1, len(worksheets)):
        #     worksheets[i].write(h_row, h_col - 1, "Average : ")
        #     worksheets[i].write(h_row, h_col, overall_performances_c[i-1]["accuracy"])
        #     worksheets[i].write(h_row, h_col + 1, overall_performances_c[i-1]["micro_precision"])
        #     worksheets[i].write(h_row, h_col + 2, overall_performances_c[i-1]["micro_recall"])
        #     worksheets[i].write(h_row, h_col + 3, overall_performances_c[i-1]["micro_f1"])
        #     worksheets[i].write(h_row, h_col + 4, overall_performances_c[i-1]["macro_precision"])
        #     worksheets[i].write(h_row, h_col + 5, overall_performances_c[i-1]["macro_recall"])
        #     worksheets[i].write(h_row, h_col + 6, overall_performances_c[i-1]["macro_f1"])
        #     worksheets[i].write(h_row, h_col + 7, overall_performances_c[i-1]["average_precision"])
        #     worksheets[i].write(h_row, h_col + 8, overall_performances_c[i-1]["coverage"])
        #     worksheets[i].write(h_row, h_col + 9, overall_performances_c[i-1]["ranking_loss"])
        #     worksheets[i].write(h_row, h_col + 10, overall_performances_c[i-1]["hamming_loss"])
        #     worksheets[i].write(h_row, h_col + 11, overall_performances_c[i-1]["cross_entropy"])

        h_row += 2
    # for i in range(len(workbooks)):
    #     workbooks[i].close()
    all_results_c[str(0)] = config
    np.save(path.join(fldr1, str.split(config.LOG_DIR, "/")[1] + '_results_c.npy'), all_results_c)
    l_res_c.append({str(0): config})
    all_avg_results_c[str.split(config.LOG_DIR, "/")[1]] = l_res_c
    fn1 = str.split(config.LOG_DIR, "/")[1] + '_results_avg_n.npy'
    fn2 = str.split(config.LOG_DIR, "/")[1] + '_results_avg_c.npy'
    np.save(path.join(fldr1, "Avg", fn2), all_avg_results_c)

if __name__ == "__main__":
    main()

