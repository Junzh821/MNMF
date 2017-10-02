import argparse
import logging
import collections
import numpy as np
from os import path, makedirs, listdir
from scipy.io import loadmat
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC

import data_utils as du
import latent_graphs as lg
import eval_performance as ep
import mnf as mnf
import mnf_sp as mnfsp




def get_ArgumentParser():
    parser = argparse.ArgumentParser()

    # Experiment configuration
    parser.add_argument("--DATA_DIR", default="citeseer")
    parser.add_argument("--LOG_DIR", default="emb/")
    parser.add_argument("--FOLDER_SUFFIX", default=" ")
    parser.add_argument("--LOG_FNAME", default="mod_mnf.log")
    parser.add_argument("--MODEL", default="data/")
    # Factorization
    parser.add_argument("--LG", default=1, help="Latent graph integration ")
    parser.add_argument("--FACT_Y", default=1, help="Factorize Label matrix")
    parser.add_argument("--ENF_Y", default=0, help="Enforce Label matrix")

    # Weights for different components in the objective functions
    parser.add_argument("--ETA", default=1.0, help="First and Second order proximity mixing parameters")
    parser.add_argument("--ALPHA", default=7.0, help="Similarity matrix factorization weight")
    parser.add_argument("--BETA", default=3.0, help="Community Indicator matrix factorization weight")
    parser.add_argument("--THETA", default=1.0, help="Label matrix factorization weight")
    parser.add_argument("--GAMMA", default=0.3, help="Modularity Maximization weight")
    parser.add_argument("--PHI", default=1.0, help="Network regularization weight")
    parser.add_argument("--DELTA", default=1.0, help="Simple Network regularization weight")
    parser.add_argument("--LAMBDA", default=1.0, help="L2 regularization weight for S and H factorization")
    parser.add_argument("--ZETA", default=1.0e+9, help="L2 regularization weight for solving H")
    # paper to code attribute mapping : alpha for S factorization, beta for H factorization, gamma for modularity maximization

    # Factorization parameters
    parser.add_argument("--MAX_ITER", default=100)
    parser.add_argument("--L_COMPONENTS", default=128)
    parser.add_argument("--K", default=5)
    parser.add_argument("--INIT", default="random")
    parser.add_argument("--PROJ", default=True)
    parser.add_argument("--COST_F", default='LS')
    parser.add_argument("--CONV_LS", default=5e-6)
    parser.add_argument("--CONV_KL", default=1e-4)
    parser.add_argument("--CONV_MUL", default=1e-4)
    parser.add_argument("--MULTI_LABEL", default=False)

    return parser

def init(dir_name, file_name):
    if not path.exists(dir_name):
        makedirs(dir_name)
    file_name = dir_name +"_"+ file_name
    logging.basicConfig(filename=file_name, filemode='w', level=logging.DEBUG)
    return logging.getLogger('MNF')

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
    # attributes = preprocessing.normalize(attributes, axis=1, norm='l2')
    relations = [loadmat(path.join(dir_name, 'Relation', name))['adjmat'] for name in relation_names]
    for idx in range(len(relations)) :
        if issparse(relations[idx]) :
            relations[idx] = relations[idx].toarray().astype(np.float64)
        # relations[idx] = preprocessing.normalize(relations[idx], axis=1, norm='l2')

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

def get_perf_metrics_using_lr(config, entity_embedding, labels, train_ids, val_ids, test_ids) :
    pred_ids = test_ids
    labelled_ids = train_ids
    clf = OneVsRestClassifier(LogisticRegression())
    entity_embedding = normalize(entity_embedding, axis=1, norm='l2')
    clf.fit(entity_embedding[labelled_ids, :], labels[labelled_ids, :])
    predictions = clf.predict_proba(entity_embedding[pred_ids, :])
    performances = ep.evaluate(predictions, labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return performances

def get_perf_metrics_using_svm(config, entity_embedding, labels, train_ids, val_ids, test_ids) :
    pred_ids = test_ids
    labelled_ids = train_ids
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    entity_embedding = normalize(entity_embedding, axis=1, norm='l2')
    clf.fit(entity_embedding[labelled_ids, :], labels[labelled_ids, :])
    predictions = clf.decision_function(entity_embedding[pred_ids, :])
    performances = ep.evaluate(predictions, labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return performances

def get_perf_metrics(config, entity_embedding, Q, labels, train_ids, val_ids, test_ids) :
    pred_ids = test_ids
    labelled_ids = train_ids
    Y_hat = np.dot(Q, entity_embedding.T)
    Y_hat = Y_hat.T
    Y_hat = normalize(Y_hat, axis=1, norm='l2')
    perf = ep.evaluate(Y_hat[pred_ids, :], labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return perf

def tune_model_using_lr(config, entity_embedding, labels, train_ids, val_ids) :
    pred_ids = val_ids
    labelled_ids = train_ids
    clf = OneVsRestClassifier(LogisticRegression())
    entity_embedding = normalize(entity_embedding, axis=1, norm='l2')
    clf.fit(entity_embedding[labelled_ids, :], labels[labelled_ids, :])
    predictions = clf.predict_proba(entity_embedding[pred_ids, :])
    performances = ep.evaluate(predictions, labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return performances

def tune_model_using_svm(config, entity_embedding, labels, train_ids, val_ids) :
    pred_ids = val_ids
    labelled_ids = train_ids
    clf = OneVsRestClassifier(LinearSVC(random_state=0))
    entity_embedding = normalize(entity_embedding, axis=1, norm='l2')
    clf.fit(entity_embedding[labelled_ids, :], labels[labelled_ids, :])
    predictions = clf.decision_function(entity_embedding[pred_ids, :])
    performances = ep.evaluate(predictions, labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return performances

def tune_model(config, entity_embedding,  Q, labels, train_ids, val_ids) :
    pred_ids = val_ids
    labelled_ids = train_ids
    Y_hat = np.dot(Q, entity_embedding.T)
    Y_hat = Y_hat.T
    Y_hat = normalize(Y_hat, axis=1, norm='l2')
    performances = ep.evaluate(Y_hat[pred_ids, :], labels[pred_ids, :], threshold=0, multi_label=config.MULTI_LABEL)
    return performances

def main():
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Matrix Factorization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    config = get_ArgumentParser().parse_args()
    if config.DATA_DIR in ['washington', 'wisconsin', 'texas', 'cornell', 'armherst', 'rochester', 'mich', 'hamilton',
                           'citeseer', 'cora', 'wiki']:
        config.MULTI_LABEL = False
    elif config.DATA_DIR in['ppi', 'blogcatalog', 'wiki_n2v']:
        config.MULTI_LABEL = True
    print("Config: %s" % (config))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Initialization and loading dataset
    logger = init(config.LOG_DIR, config.LOG_FNAME)
    dataset = load_dataset(path.join("../Datasets/", config.DATA_DIR.lower()))
    tmp = {"accuracy": 0, "micro_precision": 0, "micro_recall": 0, "micro_f1": 0, "macro_precision": 0,
                   "macro_recall": 0, "macro_f1": 0, "average_precision": 0, "coverage": 0, "ranking_loss": 0,
                   "hamming_loss": 0, "cross_entropy": 0, "bae": 0, "pak": 0}
    overall_performances_a = [tmp]
    overall_performances_b = [tmp]
    overall_performances_c = [tmp]
    all_results_a = {}
    all_avg_results_a = {}
    all_results_b = {}
    all_avg_results_b = {}
    all_results_c = {}
    all_avg_results_c = {}
    l_res_a = list()
    l_res_b = list()
    l_res_c = list()

    # graph_file = path.join(config.MODEL, "Net", config.DATA_DIR.title() + "_net.txt")
    # S = np.loadtxt(graph_file)
    # S = du.get_proximity_similarity_matrix(dataset.relations[0], float(config.ETA))
    # S = csr_matrix(du.get_proximity_matrix(dataset.relations[0], float(config.ETA)))
    # B = csr_matrix(du.get_modularity_matrix(dataset.relations[0]))
    S = du.get_proximity_matrix(dataset.relations[0], float(config.ETA))
    B = du.get_modularity_matrix(dataset.relations[0])

    perc_data = dataset.expt_sets
    for a in perc_data :
        all_results_a[a] = {}
        all_results_b[a] = {}
        all_results_c[a] = {}
        all_avg_results_a[config.FOLDER_SUFFIX] = list()
        all_avg_results_b[config.FOLDER_SUFFIX] = list()
        all_avg_results_c[config.FOLDER_SUFFIX] = list()
        overall_performances_a = [dict.fromkeys(g, 0) for g in overall_performances_a]
        overall_performances_b = [dict.fromkeys(g, 0) for g in overall_performances_b]
        overall_performances_c = [dict.fromkeys(g, 0) for g in overall_performances_c]
        itr = 0
        print("% of randomly sampled training data ---- ", a)
        # for b in range(1, dataset.n_folds+1) :
        for b in range(1, 2):
            data_dir = path.join("../Datasets/", config.DATA_DIR.lower(), 'index', str(a), str(b))
            train_ids = np.load(path.join(data_dir, 'train_ids.npy')).astype(dtype=bool)
            val_ids = np.load(path.join(data_dir, 'val_ids.npy')).astype(dtype=bool)
            train_ids = np.logical_or(train_ids, val_ids)
            test_ids = np.load(path.join(data_dir, 'test_ids.npy')).astype(dtype=bool)
            # test_ids = np.logical_or(test_ids, val_ids)

            labelled_ids = train_ids
            unlabelled_ids = np.logical_not(labelled_ids)
            n_unlabelled = np.count_nonzero(unlabelled_ids)
            n_labelled = np.count_nonzero(labelled_ids)
            labels = np.copy(dataset.truth)
            labels[unlabelled_ids, :] = np.zeros((n_unlabelled, dataset.n_labels))
            # Y = csr_matrix(dataset.truth)
            # Y_train = csr_matrix(labels)
            # D = [csr_matrix(i.T) for i in dataset.attributes]  # mxn
            # X = [csr_matrix(i) for i in dataset.relations]  # nxn
            Y = dataset.truth
            Y_train = labels
            D = [i.T for i in dataset.attributes]  # mxn
            X = [i for i in dataset.relations]  # nxn

            performances_a = []
            performances_b = []
            performances_c = []
            best_result_lr, best_result_svm, best_result = mnf.factorize(config, dataset, S, B, D[0], X[0], Y.T, Y_train.T,
                                                                          train_ids, val_ids, test_ids, logger)


            # outputEntities = path.join(config.LOG_DIR, "U_" + str(a) + "_" + str(b) + "_" + "_n.log")  # U
            # np.savetxt(outputEntities, best_result_n['U'], fmt="%f")
            # outputEntities = path.join(config.LOG_DIR, "H_" + str(a) + "_" + str(b) + "_" + "_n.log")  # U
            # np.savetxt(outputEntities, best_result_n['H'], fmt="%f")
            # outputEntities = path.join(config.LOG_DIR, "Q_" + str(a) + "_" + str(b) + "_" + "_n.log")  # U
            # np.savetxt(outputEntities, best_result_n['Q'], fmt="%f")

            # performance_lr = get_perf_metrics_using_lr(config, best_result_lr['U'], Y.toarray(), train_ids, val_ids,
            #                                            test_ids)
            # print("Performance_using_LR : Test accuracy: {%0.5f } , Test Loss: {%0.5f } Iter: {%d}" % (
            #     performance_lr['accuracy'], performance_lr['cross_entropy'], best_result_lr['i']))
            # performances_a.append(performance_lr)
            # performance_svm = get_perf_metrics_using_svm(config, best_result_svm['U'], Y.toarray(), train_ids, val_ids,
            #                                              test_ids)
            # print("Performance_using_SVM : Test accuracy: {%0.5f } , Test Loss: {%0.5f } Iter: {%d}" % (
            #     performance_svm['accuracy'], performance_svm['cross_entropy'], best_result_svm['i']))
            # performances_b.append(performance_svm)
            # performance = get_perf_metrics(config, best_result['U'], best_result['Q'], Y.toarray(), train_ids, val_ids,
            #                                test_ids)
            # print("Performance_without_classifier : Test accuracy: {%0.5f } , Test Loss: {%0.5f } Iter: {%d}" % (
            #     performance['accuracy'], performance['cross_entropy'], best_result['i']))
            performance_lr = get_perf_metrics_using_lr(config, best_result_lr['U'], Y, train_ids, val_ids,
                                                       test_ids)
            print("Performance_using_LR : Test accuracy: {%0.5f } , Test Loss: {%0.5f } Iter: {%d}" % (
                performance_lr['accuracy'], performance_lr['cross_entropy'], best_result_lr['i']))
            performances_a.append(performance_lr)
            performance_svm = get_perf_metrics_using_svm(config, best_result_svm['U'], Y, train_ids, val_ids,
                                                         test_ids)
            print("Performance_using_SVM : Test accuracy: {%0.5f } , Test Loss: {%0.5f } Iter: {%d}" % (
                performance_svm['accuracy'], performance_svm['cross_entropy'], best_result_svm['i']))
            performances_b.append(performance_svm)
            performance = get_perf_metrics(config, best_result['U'], best_result['Q'], Y, train_ids, val_ids,
                                           test_ids)
            print("Performance_without_classifier : Test accuracy: {%0.5f } , Test Loss: {%0.5f } Iter: {%d}" % (
                performance['accuracy'], performance['cross_entropy'], best_result['i']))
            performances_c.append(performance)
            all_results_a[a][b] = performance_lr
            all_results_b[a][b] = performance_svm
            all_results_c[a][b] = performance

            for i in range(len(overall_performances_a)):
                if len(overall_performances_a) == len(performances_a):
                    overall_performances_a[i]["accuracy"] += performances_a[i]["accuracy"]
                    overall_performances_a[i]["micro_precision"] += performances_a[i]["micro_precision"]
                    overall_performances_a[i]["micro_recall"] += performances_a[i]["micro_recall"]
                    overall_performances_a[i]["micro_f1"] += performances_a[i]["micro_f1"]
                    overall_performances_a[i]["macro_precision"] += performances_a[i]["macro_precision"]
                    overall_performances_a[i]["macro_recall"] += performances_a[i]["macro_recall"]
                    overall_performances_a[i]["macro_f1"] += performances_a[i]["macro_f1"]
                    overall_performances_a[i]["average_precision"] += performances_a[i]["average_precision"]
                    overall_performances_a[i]["coverage"] += performances_a[i]["coverage"]
                    overall_performances_a[i]["ranking_loss"] += performances_a[i]["ranking_loss"]
                    overall_performances_a[i]["hamming_loss"] += performances_a[i]["hamming_loss"]
                    overall_performances_a[i]["cross_entropy"] += performances_a[i]["cross_entropy"]
                    overall_performances_a[i]["bae"] += performances_a[i]["bae"]
                    overall_performances_a[i]["pak"] += performances_a[i]["pak"]
            for i in range(len(overall_performances_b)):
                if len(overall_performances_b) == len(performances_b):
                    overall_performances_b[i]["accuracy"] += performances_b[i]["accuracy"]
                    overall_performances_b[i]["micro_precision"] += performances_b[i]["micro_precision"]
                    overall_performances_b[i]["micro_recall"] += performances_b[i]["micro_recall"]
                    overall_performances_b[i]["micro_f1"] += performances_b[i]["micro_f1"]
                    overall_performances_b[i]["macro_precision"] += performances_b[i]["macro_precision"]
                    overall_performances_b[i]["macro_recall"] += performances_b[i]["macro_recall"]
                    overall_performances_b[i]["macro_f1"] += performances_b[i]["macro_f1"]
                    overall_performances_b[i]["average_precision"] += performances_b[i]["average_precision"]
                    overall_performances_b[i]["coverage"] += performances_b[i]["coverage"]
                    overall_performances_b[i]["ranking_loss"] += performances_b[i]["ranking_loss"]
                    overall_performances_b[i]["hamming_loss"] += performances_b[i]["hamming_loss"]
                    overall_performances_b[i]["cross_entropy"] += performances_b[i]["cross_entropy"]
                    overall_performances_b[i]["bae"] += performances_b[i]["bae"]
                    overall_performances_b[i]["pak"] += performances_b[i]["pak"]
            for i in range(len(overall_performances_c)):
                if len(overall_performances_c) == len(performances_c):
                    overall_performances_c[i]["accuracy"] += performances_c[i]["accuracy"]
                    overall_performances_c[i]["micro_precision"] += performances_c[i]["micro_precision"]
                    overall_performances_c[i]["micro_recall"] += performances_c[i]["micro_recall"]
                    overall_performances_c[i]["micro_f1"] += performances_c[i]["micro_f1"]
                    overall_performances_c[i]["macro_precision"] += performances_c[i]["macro_precision"]
                    overall_performances_c[i]["macro_recall"] += performances_c[i]["macro_recall"]
                    overall_performances_c[i]["macro_f1"] += performances_c[i]["macro_f1"]
                    overall_performances_c[i]["average_precision"] += performances_c[i]["average_precision"]
                    overall_performances_c[i]["coverage"] += performances_c[i]["coverage"]
                    overall_performances_c[i]["ranking_loss"] += performances_c[i]["ranking_loss"]
                    overall_performances_c[i]["hamming_loss"] += performances_c[i]["hamming_loss"]
                    overall_performances_c[i]["cross_entropy"] += performances_c[i]["cross_entropy"]
                    overall_performances_c[i]["bae"] += performances_c[i]["bae"]
                    overall_performances_c[i]["pak"] += performances_c[i]["pak"]
            print("**********************************************************")
            itr += 1

        overall_performances_a = [{k: v / dataset.n_folds for k, v in d.items()} for d in
                                  overall_performances_a]
        overall_performances_b = [{k: v / dataset.n_folds for k, v in d.items()} for d in
                                  overall_performances_b]
        overall_performances_c = [{k: v / dataset.n_folds for k, v in d.items()} for d in
                                  overall_performances_c]
        print('LR ---> ', overall_performances_a)
        print('SVM ---> ', overall_performances_b)
        print('N ---> ', overall_performances_c)
        l_res_a.append({a: overall_performances_a[0]})
        l_res_b.append({a: overall_performances_b[0]})
        l_res_c.append({a: overall_performances_c[0]})

    all_results_a[str(0)] = config
    all_results_b[str(0)] = config
    all_results_c[str(0)] = config
    np.save(path.join(config.LOG_DIR, 'results_lr.npy'), all_results_a)
    np.save(path.join(config.LOG_DIR, 'results_svm.npy'), all_results_b)
    np.save(path.join(config.LOG_DIR, 'results_n.npy'), all_results_c)
    l_res_a.append({str(0): config})
    l_res_b.append({str(0): config})
    l_res_c.append({str(0): config})
    all_avg_results_a[config.FOLDER_SUFFIX] = l_res_a
    all_avg_results_b[config.FOLDER_SUFFIX] = l_res_b
    all_avg_results_c[config.FOLDER_SUFFIX] = l_res_c
    fn = path.join(config.LOG_DIR, "Avg")
    if not path.exists(fn):
        makedirs(fn, exist_ok=True)
    np.save(path.join(config.LOG_DIR, "Avg", 'results_avg_lr.npy'), all_avg_results_a)
    np.save(path.join(config.LOG_DIR, "Avg", 'results_avg_svm.npy'), all_avg_results_b)
    np.save(path.join(config.LOG_DIR, "Avg", 'results_avg_n.npy'), all_avg_results_c)

if __name__ == "__main__":
    main()

