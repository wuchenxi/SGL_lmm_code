import csv
import os
import scipy as SP

root = 'data'
simu_chr1_file = os.path.join(root, 'simu_chr1.csv')
output_file = os.path.join(root, 'simu_chr1_01.csv')
maf_file = os.path.join(root, 'simu_chr1_maf.csv')

def normalize(l):
    count={'A':0,'T':0,'G':0,'C':0}
    for c in l:
        count[c]+=1
    dft=max(count,key=count.get)
    l = SP.array(l)
    arr = SP.array(l!=dft, SP.uint8)
    return arr, arr.mean()


data = list()
with open(simu_chr1_file, 'r') as f:
	reader = csv.reader(f)
	title1 = reader.next();
	title2 = reader.next();
	for line in reader:
		data.append(line)

res = list()
maf = list()
for line in data:
	na, a = normalize(line[2:])
	if a > 0.05:
		res.append([line[0],line[1]]+na.tolist())	
		maf.append(a)


with open(output_file, 'w') as f:
	writer = csv.writer(f, quoting=csv.QUOTE_NONE)
	for line in res:
		writer.writerow(line)

with open(maf_file, 'w') as f:
	writer = csv.writer(f, quoting=csv.QUOTE_NONE)
	for line in maf:
		writer.writerow([line])
