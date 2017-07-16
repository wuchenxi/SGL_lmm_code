import csv
src_file = 'call_method_75_TAIR9.csv'
simu_chr1_file = 'simu_chr1.csv'
simu_chr2_5_file = 'simu_chr2_5.csv'

simu_chr1 = list()
simu_chr2_5 = list()
with open(src_file, 'rb') as f:
	reader = csv.reader(f)
	title1 = reader.next()
	title2 = reader.next()
	i = 0
	for line in reader:
		i += 1
		if i % 50000 == 0: print i
		if line[0] == '1': simu_chr1.append(line)
		else: simu_chr2_5.append(line)	


print ('write chr1 now..')
with open(simu_chr1_file, 'wb') as f:
	writer = csv.writer(f, quoting=csv.QUOTE_NONE)
	writer.writerow(title1)
	writer.writerow(title2)
	for line in simu_chr1:
		writer.writerow(line)


print ('write chr2_5 now..')
with open(simu_chr2_5_file, 'wb') as f:
	writer = csv.writer(f, quoting=csv.QUOTE_NONE)
	writer.writerow(title1)
	writer.writerow(title2)
	for line in simu_chr2_5:
		writer.writerow(line)

print ('Done')
