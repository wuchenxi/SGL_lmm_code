"""
"""

import csv
import scipy as SP
import scipy.linalg as LA
import pdb
import lmm_lasso_pg as lmm_lasso
import os
import sys
from pandas_plink import read_plink

def get_group_idx(group_file):
    results = []
    file = open(group_file)
    lines = file.readlines()
    for line in lines:
        info = line.split()
        if 'gene' in info:
            results.append([info[3], info[4]])
    return SP.array(results).astype(int)

def get_index(bim):
    pos = list(bim['pos'])
    index = [idx for idx in pos if idx != 0]
    return SP.array(index).astype(int)

def process_group(group_idx, idx):
    n = len(group_idx)
    total_idx = [[group_idx[i, 0], 'left-bound', i] for i in range(n)]

    #Combination1
    i = j = 0
    while j < n:
        while i < n and group_idx[i, 0] <= group_idx[j, 1]:
            i += 1
        if i < n:
            total_idx.insert(i + j, [group_idx[j, 1], 'right-bound', j])
        else:
            total_idx.append([group_idx[j, 1], 'right-bound', j])
        j += 1

    #Combination2
    i = j = 0
    n = len(total_idx)
    m = len(idx)
    while j < m:
        while i < n and total_idx[i][0] <= idx[j]:
            i += 1
        if i < n:
            total_idx.insert(i + j, [idx[j], 'point'])
        else:
            total_idx.append([idx[j], 'point'])
        j += 1

    cur_inv = {}
    results = []
    for item in total_idx:
        if item[1] == 'left-bound':
            cur_inv[item[2]] = [0x7fffffff, 0]
        elif item[1] == 'point':
            for k in cur_inv.keys():
                cur_inv[k] = [min(cur_inv[k][0], item[0]), max(cur_inv[k][1], item[0])]
        elif item[1] == 'right-bound':
            if cur_inv.has_key(item[2]):
                results.append(cur_inv[item[2]])
                cur_inv.pop(item[2])

    return SP.array(results).astype(int)



if __name__ == '__main__':

    if len(sys.argv) < 5:
        sys.stderr.write('\tUsage: python test_cv_true.py [gene_file] [kinship_file] [use_group_lasso] [use_lasso]\n');
        sys.exit(1);

    if sys.argv[3].lower() == 'true':
        use_group_lasso = True
    elif sys.argv[3].lower() == 'false':
        use_group_lasso = False
    else:
        sys.stderr.write('invalid use_group_lasso value: true of false')

    if sys.argv[4].lower() == 'true':
        use_lasso = True
    elif sys.argv[4].lower() == 'false':
        use_lasso = False
    else:
        sys.stderr.write('invalid use_lasso value: true of false')

    root = 'data'
    gene_file = os.path.join(root, sys.argv[1])
    kinship_file = os.path.join(root, sys.argv[2])
    group_file = os.path.join(root, "gencode.v25lift37.basic.annotation.gtf")

    # load genotypes
    [bim, fam, G] = read_plink(gene_file)

    X = SP.array(G.compute()).astype(float)

    [n_f, n_s] = X.shape
    for i in range(n_f):
        m = X[i].mean()
        std = X[i].std()
        X[i] = (X[i] - m) / std
        print len(X[i])
    X = X.T

    # simulate phenotype
    y = SP.array(list(fam['i'])).astype(float)

    # init
    debug = False
    n_train = int(n_s * 0.7)
    n_test = n_s - n_train
    n_reps = 10
    f_subset = 0.7

    muinit = 0.1
    mu2init = 0.1
    ps_step = 3

    # split into training and testing
    train_idx = SP.random.permutation(SP.arange(n_s))
    test_idx = train_idx[n_train:]
    train_idx = train_idx[:n_train]

    # calculate kernel
    # the first 2622 SNP are in the first chromosome which we are testing
    [kinship_bim, kinship_fam, kinship_G] = read_plink(kinship_file)
    XO = SP.array(kinship_G.compute()).astype(float)
    K = 1. / XO.shape[0] * SP.dot(XO.T, XO)

    ################################################
    #  :Get group
    #    group = SP.array(list(csv.reader(open(group_file, 'rb'),
    #                                 delimiter=','))).astype(int)
    #    group = SP.squeeze(group, axis=1)
    #    idx = 0
    #    gp = list()
    #    for i in range(len(group)):
    #        gp.append([idx, idx + group[i]])
    #        idx += group[i]
    #    group = gp
    ################################################

    group_idx = get_group_idx(group_file)
    idx = get_index(bim)
    group = process_group(group_idx, idx)

    # Glasso Parameter selection by 5 fold cv
    if use_group_lasso:
        optmu = muinit
        optmu2 = mu2init
        optcor = 0
        for j1 in range(7):
            for j2 in range(7):
                mu = muinit * (ps_step ** j1)
                mu2 = mu2init * (ps_step ** j2)
                cor = 0
                for k in range(5):  # 5 for full 5 fold CV
                    train1_idx = SP.concatenate(
                        (train_idx[:int(n_train * k * 0.2)], train_idx[int(n_train * (k + 1) * 0.2):n_train]))
                    valid_idx = train_idx[int(n_train * k * 0.2):int(n_train * (k + 1) * 0.2)]
                    res1 = lmm_lasso.train(X[train1_idx], K[train1_idx][:, train1_idx], y[train1_idx], mu, mu2, group)
                    w1 = res1['weights']
                    yhat = lmm_lasso.predict(y[train1_idx], X[train1_idx, :], X[valid_idx, :], K[train1_idx][:, train1_idx],
                                             K[valid_idx][:, train1_idx], res1['ldelta0'], w1)
                    cor += SP.dot(yhat.T - yhat.mean(), y[valid_idx] - y[valid_idx].mean()) / (
                    yhat.std() * y[valid_idx].std())

                print mu, mu2, cor[0, 0]
                if cor > optcor:
                    optcor = cor
                    optmu = mu
                    optmu2 = mu2

        print optmu, optmu2, optcor[0, 0]

        # train
        res = lmm_lasso.train(X[train_idx], K[train_idx][:, train_idx], y[train_idx], optmu, optmu2, group)
        w = res['weights']

        # predict
        ldelta0 = res['ldelta0']
        yhat = lmm_lasso.predict(y[train_idx], X[train_idx, :], X[test_idx, :], K[train_idx][:, train_idx],
                                 K[test_idx][:, train_idx], ldelta0, w)
        corr = 1. / n_test * SP.dot(yhat.T - yhat.mean(), y[test_idx] - y[test_idx].mean()) / (
        yhat.std() * y[test_idx].std())
        print corr[0, 0]

        ss = lmm_lasso.stability_selection(X, K, y, optmu, optmu2, group, n_reps, f_subset)

        result_ss = [(ss[idx], idx) for idx in xrange(len(ss))]
        result_ss.sort(key=lambda item: (-item[0], item[1]))
        with open(gene_file + '_GroupL_result.csv', 'w') as result_file:
            result_writer = csv.writer(result_file)
            for item in result_ss:
                result_writer.writerow((item[1], item[0]))

    # lasso parameter selection by 5 fold cv
    if use_lasso:
        optmu0 = muinit
        optcor = 0
        for j1 in range(7):
            mu = muinit * (ps_step ** j1)
            cor = 0
            for k in range(5):
                train1_idx = SP.concatenate((train_idx[:int(n_train * k * 0.2)],
                                             train_idx[int(n_train * (k + 1) * 0.2):n_train]))
                valid_idx = train_idx[int(n_train * k * 0.2):int(n_train * (k + 1) * 0.2)]
                res1 = lmm_lasso.train(X[train1_idx], K[train1_idx][:, train1_idx], y[train1_idx], mu, 0, [])
                w1 = res1['weights']
                yhat = lmm_lasso.predict(y[train1_idx], X[train1_idx, :], X[valid_idx, :], K[train1_idx][:, train1_idx],
                                         K[valid_idx][:, train1_idx], res1['ldelta0'], w1)
                cor += SP.dot(yhat.T - yhat.mean(), y[valid_idx] - y[valid_idx].mean()) / (yhat.std() * y[valid_idx].std())

            print mu, cor[0, 0]
            if cor > optcor:
                optcor = cor
                optmu0 = mu

        print optmu0, optcor[0, 0]

        # train
        res = lmm_lasso.train(X[train_idx], K[train_idx][:, train_idx], y[train_idx], optmu0, 0, [])
        w = res['weights']

        # predict
        ldelta0 = res['ldelta0']
        yhat = lmm_lasso.predict(y[train_idx], X[train_idx, :], X[test_idx, :], K[train_idx][:, train_idx],
                                 K[test_idx][:, train_idx], ldelta0, w)
        corr = 1. / n_test * SP.dot(yhat.T - yhat.mean(), y[test_idx] - y[test_idx].mean()) / (
        yhat.std() * y[test_idx].std())
        print corr[0, 0]

        ss2 = lmm_lasso.stability_selection(X, K, y, optmu0, 0, [], n_reps, f_subset)

        result_ss2 = [(ss2[idx], idx) for idx in xrange(len(ss2))]
        result_ss2.sort(key=lambda item: (-item[0], item[1]))
        with open(gene_file + '_L_result.csv', 'w') as result_file:
            result_writer = csv.writer(result_file)
            for item in result_ss2:
                result_writer.writerow((item[1], item[0]))
