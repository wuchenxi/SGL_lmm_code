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
        if 'gene' in info and 'chr1' in info:
            results.append([info[3], info[4]])
    results = SP.array(results).astype(int)
    return results


def get_index(bim):
    pos = list(bim['pos'])
    index = [idx for idx in pos if idx != 0]
    return SP.array(index).astype(int)


def process_group(group_idx, idx):
    INF = 0x7fffffffff
    n = len(group_idx)
    total_idx = [[group_idx[i, 0], 'left-bound', i] for i in range(n)]

    # Combination1

    i = j = 0
    while j < n:
        while i < n and group_idx[i, 0] <= group_idx[j, 1]:
            i += 1
        if i < n:
            total_idx.insert(i + j, [group_idx[j, 1], 'right-bound', j])
        else:
            total_idx.append([group_idx[j, 1], 'right-bound', j])
        j += 1

    # Combination2

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
            cur_inv[item[2]] = [INF, 0]
        elif item[1] == 'point':
            if len(cur_inv) == 0:
                results.append([item[0], item[0]])
            else:
                for k in cur_inv.keys():
                    cur_inv[k] = [min(cur_inv[k][0], item[0]), max(cur_inv[k][1], item[0])]
        elif item[1] == 'right-bound':
            if cur_inv.has_key(item[2]):
                if cur_inv[item[2]][0] < INF and cur_inv[item[2]][1] > 0 and not cur_inv[item[2]] in results:
                    results.append(cur_inv[item[2]])
                cur_inv.pop(item[2])
    results.sort()

    return SP.array(results).astype(int)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write('\tUsage: python group.py [gene_file] [group_file]\n');
        sys.exit(1);

    root = 'data'
    [bim, fam, G] = read_plink(os.path.join(root, sys.argv[1]))
    group_file = os.path.join(root, sys.argv[2])
    group_idx = get_group_idx(group_file)
    idx = get_index(bim)
    group = process_group(group_idx, idx)
    with open(os.path.join(root, sys.argv[1] + '_group.info'), 'w') as info_file:
        writer = csv.writer(info_file)
        for info in group:
            writer.writerow(info)


