"""
"""

import csv
import scipy as SP
import scipy.linalg as LA
import pdb
import lmm_lasso_pg as lmm_lasso
import os
import sys
#from pandas_plink import read_plink

if __name__ == '__main__':

    if len(sys.argv) < 4:
        sys.stderr.write('\tUsage: python test_cv_0.py [path_of_geno] [use_group_lasso] [use_lasso]\n');
        sys.exit(1);

    if sys.argv[2].lower() == 'true':
        use_group_lasso = True
    elif sys.argv[2].lower() == 'false':
        use_group_lasso = False
    else:
        sys.stderr.write('invalid use_group_lasso value: true of false')

    if sys.argv[3].lower() == 'true':
        use_lasso = True
    elif sys.argv[3].lower() == 'false':
        use_lasso = False
    else:
        sys.stderr.write('invalid use_lasso value: true of false')

    root = 'data'

    gene_file = os.path.join(root, 'gene_chr1.csv')
    ypheno_file = os.path.join(root, sys.argv[1])
    kinship_file = os.path.join(root, 'kinship_x.csv')
    group_file = os.path.join(root, 'simu_chr1_group_stru_01.csv')

    # load genotypes
    X = SP.array(list(csv.reader(open(gene_file,'rb'),
                                 delimiter=','))).astype(float)

    [n_f, n_s] = X.shape
    for i in xrange(X.shape[0]):
        m=X[i].mean()
        std=X[i].std()
        X[i]=(X[i]-m)/std
    X = X.T


    # simulate phenotype
    y = SP.array(list(csv.reader(open(ypheno_file, 'rb'),
                                delimiter=','))).astype(float) 


    # init
    debug = False

    muinit = 30
    mu2 = 0.1
    ps_step = 0.9
    n_reps=5

    # calculate kernel
    # the first 2622 SNP are in the first chromosome which we are testing
    XO = SP.array(list(csv.reader(open(kinship_file, 'rb'),
                                delimiter=','))).astype(float) 
    K = 1./XO.shape[0]*SP.dot(XO.T,XO)

    group = SP.array(list(csv.reader(open(group_file, 'rb'),
                                delimiter=','))).astype(int)
    group = SP.squeeze(group, axis=1)
    idx = 0
    gp = list()
    for i in range(len(group)):
        gp.append( [idx, idx+group[i]] )
        idx += group[i]
    group = gp

    if use_lasso:
        freq=SP.zeros(n_f)
        weight=SP.zeros(n_f)
        for k in range(20):
            opterr=10000
            optw=0
            stop=2
            w0=0
            nm={}
            mu=muinit
            train_idx = SP.random.permutation(SP.arange(n_s))
            for j in range(20):
                mu*=ps_step
                train1_idx=train_idx[:int(n_s*0.8)]
                valid_idx=train_idx[int(n_s*0.8):]
                res=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,0,[],w0=w0,null_model=nm)
                w0=res['weights']
                nm=res['null_model']
                yhat = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[valid_idx,:],K[train1_idx][:,train1_idx],K[valid_idx][:,train1_idx],res['ldelta0'],w0)
                err = LA.norm(yhat-y[valid_idx])
                print mu, err
                if err<opterr:
                    optw=w0
                    opterr=err
                    stop=2
                else:
                    stop-=1
                if stop<=0:
                    break
            freq[(SP.nonzero(optw)[0])]+=1
            weight=weight+SP.absolute(optw)
        err1=0
        err2=0
        for i in xrange(n_f):
            if i in idx:
                if freq[i]<n_reps-1:
                    err1+=1
                    print "FN", i, freq[i]
            else:
                if freq[i]>=n_reps-1:
                    err2+=1
                    print "FP", i, freq[i]
        
        result_ss = [(idx, freq[idx], weight[idx]) for idx in xrange(n_f)]
        result_ss.sort(key=lambda item: (-item[1], list(-item[2]), item[0]))
        with open(ypheno_file.replace('.csv', '_L_result.csv'), 'w') as result_file:
            result_writer = csv.writer(result_file)
            for item in result_ss:
                result_writer.writerow((item[0], item[1]))

    if use_group_lasso:
        freq=SP.zeros(n_f)
        weight=SP.zeros(n_f)
        for k in range(20):
            opterr=10000
            optw=0
            stop=2
            w0=0
            nm={}
            mu=muinit
            train_idx = SP.random.permutation(SP.arange(n_s))
            for j in range(20):
                mu*=ps_step
                train1_idx=train_idx[:int(n_s*0.8)]
                valid_idx=train_idx[int(n_s*0.8):]
                res=lmm_lasso.train(X[train1_idx],K[train1_idx][:,train1_idx],y[train1_idx],mu,mu2,group,w0=w0,null_model=nm)
                w0=res['weights']
                nm=res['null_model']
                yhat = lmm_lasso.predict(y[train1_idx],X[train1_idx,:],X[valid_idx,:],K[train1_idx][:,train1_idx],K[valid_idx][:,train1_idx],res['ldelta0'],w0)
                err = LA.norm(yhat-y[valid_idx])
                print mu, err
                if err<opterr:
                    optw=w0
                    opterr=err
                    stop=2
                else:
                    stop-=1
                if stop<=0:
                    break
            freq[(SP.nonzero(optw)[0])]+=1
            weight=weight+SP.absolute(optw)
        err1=0
        err2=0
        for i in xrange(n_f):
            if i in idx:
                if freq[i]<n_reps-1:
                    err1+=1
                    print "FN", i, freq[i]
            else:
                if freq[i]>=n_reps-1:
                    err2+=1
                    print "FP", i, freq[i]
        
        result_ss = [(idx, freq[idx], weight[idx]) for idx in xrange(n_f)]
        result_ss.sort(key=lambda item: (-item[1], list(-item[2]), item[0]))
        with open(ypheno_file.replace('.csv', '_L_result.csv'), 'w') as result_file:
            result_writer = csv.writer(result_file)
            for item in result_ss:
                result_writer.writerow((item[0], item[1]))
    
