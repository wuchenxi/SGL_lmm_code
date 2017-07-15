import csv
import os
import scipy as SP

root = 'data'
simu_chr_file = os.path.join(root, 'simu_chr2_5.csv')
output_file = os.path.join(root, 'kinship_x.csv')

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

# sampling
res = SP.array(res)
print (res.shape)
n_f = res.shape[0]
idx = range(0,n_f,2000)
res = res[idx, :]
print (res.shape)

with open(output_file, 'w') as f:
	writer = csv.writer(f, quoting=csv.QUOTE_NONE)
	for line in res:
		writer.writerow(line)

