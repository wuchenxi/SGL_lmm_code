import csv
import os
import scipy as SP
from sklearn.decomposition import PCA

root = 'data'
simu_chr_file = os.path.join(root, 'simu_chr2_5.csv')
output_file = os.path.join(root, 'pca_x.csv')

def normalize(l):
    count={'A':0,'T':0,'G':0,'C':0}
    for c in l:
        count[c]+=1
    dft=max(count,key=count.get)
    l = SP.array(l)
    arr = SP.array(l!=dft, SP.uint8)
    return arr, arr.mean()


data = list()
with open(simu_chr_file, 'r') as f:
	reader = csv.reader(f)
	title1 = reader.next();
	title2 = reader.next();
	for line in reader:
		data.append(line[2:])

res = list()
for line in data:
	na, a = normalize(line)
	if a > 0.05:
		res.append(na)	

res = SP.array(res)
res = res.T
res = PCA(10).fit_transform(res)

with open(output_file, 'w') as f:
	writer = csv.writer(f, quoting=csv.QUOTE_NONE)
	for line in res:
		writer.writerow(line)

