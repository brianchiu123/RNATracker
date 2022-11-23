import csv
import gzip
import collections
with open('try.csv', newline='') as f:
	reader = csv.reader(f)
	data = list(reader)
#print(data)

del data[0]
LOCALIZATION = [
	"ER_membrane",
	"Nuclear_lamina",
	"Mito_matrix",
	"Cytosol",
	"Nucleolus",
	"Nucleus",
	"Nuclear_pore",
	"Outer_mito_membrane"
]
total = []

for i in data :
	temp=[i[0]]
	Name=0
	for j in range(4,26,3) :
		if i[j+1] == "" :
			i[j+1]= 1
		if i[j+2] == "" :
			i[j+2] = 0
		if float(i[j+1]) >0 and float(i[j+2]) < 0.05    :
			temp.append(LOCALIZATION[Name])
		Name+=1
	total.append(temp)

fname= "Homo_sapiens.GRCh38.90.gtf.gz"
retval = collections.defaultdict(list)
gene_info=[]
opener = gzip.open if fname.endswith(".gz") else open
with opener(fname) as source:
	for line in source:
		temp=[]
		line = line.decode('utf8')
		if line.startswith("#"):
			continue
		tokens = line.strip().split("\t")
		chrom, source, feature, start, end, score, strand, frame, attr = tokens

		if feature == "gene" :

			attr_split = [chunk.strip().split(" ", 1) for chunk in attr.split(";") if chunk]
			attr_dict = {k: v.strip('"') for k, v in attr_split}
			gene_id = attr_dict['gene_id']


			for i in range(len(total)):

				if total[i][0][0:15] == gene_id:
					total[i].insert(1,end)
					total[i].insert(1,start)
					total[i].insert(1,chrom)
					break
CHROMOSOMES = [str(i + 1) for i in range(22)] + ['X', 'Y']

for i in range(len(total)):
	if total[i][1] not in CHROMOSOMES : 
		total[i].insert(1,"0")
		total[i].insert(1,"0")
		total[i].insert(1,"null")

print(total)
# # Sanity check output
# for trans_id, coords in retval.items():
#     assert len(set([c[1] for c in coords])) == 1, f"Got multiple chroms for {trans_id}"
#     assert len(set([c[4] for c in coords])) == 1, f"Got multiple strands for {trans_id}"
#     assert len(set([c[-1] for c in coords])) == 1, f"Got multiple chroms for {trans_id}"
#     assert max([c[0] for c in coords]) == len(coords)  # No extra exons
#     retval[trans_id] = sorted(coords)  # Ensure ordering of exons






f = open('news.txt','w')
for i in total :
	s=""
	for j in range(len(i)):
		s=s+i[j]+" "
	f.write(s+"\n")