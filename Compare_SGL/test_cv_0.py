"""
"""

import csv
import scipy as SP
import scipy.linalg as LA
import pdb
import lmm_lasso_pg as lmm_lasso
import os
import sys

if __name__ == '__main__':

    if len(sys.argv) < 3:
        sys.stderr.write('\tUsage: python test_cv.py [ypheno.csv] [has_pca]\n')
        sys.exit(1)

    if sys.argv[2].lower() == 'true':
        has_pca = True
    elif sys.argv[2].lower() == 'false':
        has_pca = False
    else:
        sys.stderr.write('invalid has_pca value: true or false')
        sys.exit(1)

    root = 'data'
    gene_file = os.path.join(root, 'gene_chr1.csv')
    ypheno_file = os.path.join(root, 'pheno', sys.argv[1])
    pca_x_file = os.path.join(root, 'pca_x.csv')
    group_file = os.path.join(root, 'simu_chr1_group_stru_01.csv')

    # load genotypes
    X = SP.array(list(csv.reader(open(gene_file, 'rb'),
                                 delimiter=','))).astype(float)

    for i in xrange(X.shape[0]):
        m = X[i].mean()
        std = X[i].std()
        X[i] = (X[i] - m) / std
    X = X.T

    if has_pca:
        pca_x = SP.array(list(csv.reader(open(pca_x_file, 'rb'),
                                         delimiter=','))).astype(float)
        X = SP.column_stack((X, pca_x))

    # simulate phenotype
    y = SP.array(list(csv.reader(open(ypheno_file, 'rb'),
                                 delimiter=','))).astype(float)

    [n_s, n_f] = X.shape

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

    group = SP.array(list(csv.reader(open(group_file, 'rb'),
                                     delimiter=','))).astype(int)
    group = SP.squeeze(group, axis=1)
    idx = 0
    gp = list()
    for i in range(len(group)):
        gp.append([idx, idx + group[i]])
        idx += group[i]
    group = gp

    if has_pca: group = group + [[n_f - 10, n_f]]

    # Glasso Parameter selection by 5 fold cv
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
                w1 = lmm_lasso.train_lasso(X[train1_idx], y[train1_idx], mu, mu2, group)
                # predict
                idx = w1.nonzero()[0]
                Xvalid = X[valid_idx, :]
                yhat = SP.dot(Xvalid[:, idx], w1[idx])
                cor += SP.dot(yhat.T - yhat.mean(), y[valid_idx] - y[valid_idx].mean()) / (
                yhat.std() * y[valid_idx].std())

            print mu, mu2, cor[0, 0]
            if cor > optcor:
                optcor = cor
                optmu = mu
                optmu2 = mu2

    print optmu, optmu2, optcor[0, 0]

    # train
    w = lmm_lasso.train_lasso(X[train_idx], y[train_idx], optmu, optmu2, group)

    # predict
    idx = w.nonzero()[0]
    Xtest = X[test_idx, :]
    yhat = SP.dot(Xtest[:, idx], w[idx])
    corr = 1. / n_test * SP.dot(yhat.T - yhat.mean(), y[test_idx] - y[test_idx].mean()) / (
    yhat.std() * y[test_idx].std())
    print corr[0, 0]

    # stability selection
    # group info included
    ss = lmm_lasso.stability_selection(X, K, y, optmu, optmu2, group, n_reps, f_subset)

    sserr1 = 0
    sserr2 = 0
    for i in range(n_f):
        if i in idx:
            if ss[i] < n_reps * 0.8:
                sserr1 += 1
        else:
            if ss[i] >= n_reps * 0.8:
                sserr2 += 1

    # Output
    result_ss = [(ss[idx], idx) for idx in len(ss)]
    result_ss.sort(key = lambda item : (-item[0], item[1]))
    with open(ypheno_file.replace('.csv', '{0}_result.csv').format('_SGL_10pc' if has_pca else '_SGL'), 'w') as result_file:
        result_writer = csv.writer(result_file)
        for item in result_ss:
            result_writer.writerow((idx, ss[idx]))

    for i in range(n_f):
        print i, (i in idx), ss[i], ss2[i]

    print optmu, optmu2, optmu0
    print sserr1, sserr2, ss2err1, ss2err2
