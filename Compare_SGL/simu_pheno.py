import csv
import os
from datetime import datetime
import numpy as np
import scipy as sp

root = 'data/pheno'

def read_csv(filename, dtype='float', is_vector=False, has_title=False):
    data = list()
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        if has_title: title = reader.next()
        for line in reader:
            if dtype == 'float':
                if is_vector: data.append(float(line[0]))
                else: data.append([ float(d) for d in line])
            elif dtype == 'int':
                if is_vector: data.append(int(line[0]))
                else: data.append([ int(d) for d in line])
            else: 
                if is_vector: data.append(line[0])
                else: data.append(line)
    if has_title: return np.array(data), title
    else: return np.array(data)


def write_csv(filename, data, is_vector=False, title=None):
    with open(filename, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE)
        if title != None: writer.writerow(title)
        for line in data:
            if is_vector: writer.writerow([line])
            else: writer.writerow(line)


def simu_pheno(K=200, sigma_pop=0.5, sigma_sig=0.7, w=1, ex = 1):
    print ('set K={0}, sigma_pop={1}, sigma_sig={2}, w={3}, ex={4}'.format(
        K, sigma_pop, sigma_sig, w, ex))
    idx_file = 'data/simu_chr1_sigSNP_index.csv'
    geno_file = 'data/gene_chr1.csv'
    pop_pheno_file = 'data/phenotype75.csv'

    X = read_csv(geno_file)
    idx = read_csv(idx_file, dtype='int', is_vector=True)
    ypop = read_csv(pop_pheno_file, is_vector=True)
    idx = idx[:K]
 
    X = X[idx, :]
    [n_f, n_s] = X.shape
    for i in xrange(n_f):
        m=X[i].mean()
        std=X[i].std()
        X[i]=(X[i]-m)/std
    X = X.T

    ysig = sp.sum(w*X,axis=1)
    ysig = sp.reshape(ysig,(n_s,1))
    ysig = (ysig-ysig.mean())/ysig.std()

    ypop = sp.reshape(ypop,(n_s,1))
    y = sigma_sig*ysig+(1-sigma_sig)*(sigma_pop*ypop+(1-sigma_pop)
        * sp.random.randn(n_s,1)) 
    y = (y-y.mean())/y.std()
    
    # write to file
    filename = os.path.join(root, 'ex{0}'.format(ex), 'ypheno_{0}_{1}_{2}_{3}.csv'.format(K,sigma_pop,sigma_sig,w))
    print ('[{}]\twrite file '.format(datetime.now())+filename+'...')  
    write_csv(filename, y)
    return y

sigma_pop_list = [0, 0.3, 0.5, 0.7, 0.9, 1]
for current_sigma_pop in sigma_pop_list:
    simu_pheno(K=40, sigma_pop = current_sigma_pop, sigma_sig = 0.5, w = 1, ex = 1)

sigma_sig_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
for current_sigma_sig in sigma_sig_list:
    simu_pheno(K=200, sigma_pop = 0.5, sigma_sig=current_sigma_sig, w=1, ex = 2)

K_list = [40, 80, 120, 160, 200]
for current_K in K_list:
    simu_pheno(K=current_K, sigma_pop=0.5, sigma_sig=0.7, w=1, ex = 3)

w_list = [1,2,3]
for current_w in w_list:
    simu_pheno(K=200, sigma_pop = 0.5, sigma_sig = 0.7, w = current_w, ex = 4)