from Bio import Entrez
import time
import mygene
Entrez.email = "105703012@nccu.edu.tw" 


def get_info(transcript_id):
	handle = Entrez.efetch(db="nucleotide", id=transcript_id, rettype="gb", retmode="xml")
	time.sleep(1)
	a = Entrez.read(handle)
	CDS_loca = "None"
	gene_type = "None"
	protien_code = "None"
	utr_3 = "None"
	utr_5 = "None"
	Gene = "None"
	total_seq =  "None"
	total_len = "None"
	try :
		# geneid = record['Entrezgene_track-info']['Gene-track']['Gene-track_geneid']
		for i in a[0]['GBSeq_feature-table'] :
			if i['GBFeature_key'] == "CDS":
				CDS_loca = i['GBFeature_location']
			for j in i['GBFeature_quals'] : 
				if j['GBQualifier_name'] == "organism" :
					gene_type = j['GBQualifier_value']
				if j['GBQualifier_name'] == "gene" :
					Gene = j['GBQualifier_value']
				if j['GBQualifier_name'] == "translation" :
					protien_code = j['GBQualifier_value']
		total_seq = a[0]['GBSeq_sequence']
		total_len = a[0]['GBSeq_length']
		gene_type = gene_type.replace(" ","_")
		if CDS_loca != "None" :

			if "join" in CDS_loca:
				print(CDS_loca)
				c = CDS_loca.replace("join("," ")
				c = c.replace(")"," ")
				c = c.replace(","," ")
				c = c.replace(" ","")
				c=c.split("..")
				utr=['','']
				utr[0]=c[0]
				utr[1]=c[-1]

			else :
				utr = CDS_loca.split("..")

			if utr[0] != "<1":
				utr_3 = total_seq[0:int(utr[0])-1]
			if int(utr[1]) != int(total_len):
				utr_5 = total_seq[int(utr[1])+1:]
	except KeyError :
		print("can't find")
	# f = open('final_data_2.txt','w')
	# f.write("> "+Gene +" "+transcript_id+" "+gene_type+"\n")
	# f.write("CDS "+CDS_loca+"\n")
	# f.write("Location"+"\n")
	# f.write(protien_code+"\n")
	# f.write(total_seq+"\n")
	# f.write(utr_5+"\n")
	# f.write(utr_3+"\n")
	# f.close()
	print(Gene)
	print(gene_type)
	print(total_len)
	print(CDS_loca)
	print(protien_code)
	print(total_seq)
	print(utr_3)
	print(utr_5)

# data = open("transcript.txt","r")

# now = 0 
# done = 52791
# for line in data :
# 	line=line.strip('\n')
# 	if now < done :
# 		now+=1
# 		continue;

# 	line_list = line.split(" ")
# 	if "_" not in line_list[2] :
# 		continue ;
# 	transcript_id = line_list[2].strip("\n")

# 	get_info(transcript_id)

# 	now+=1
# 	print(str(now)+" "+transcript_id)
#print(get_info('NM_001170779'))
# GBQualifier_name': 'db_xref', 'GBQualifier_value': 'GeneID:159090'
handle = Entrez.efetch(db="nucleotide", id='NM_001170780', rettype="gb", retmode="xml")
a = Entrez.read(handle)
for i in a[0]['GBSeq_feature-table'] :
	for j in i['GBFeature_quals'] : 
		if j['GBQualifier_name'] == "db_xref" :
			if "GeneID" in j['GBQualifier_value'] :
				gene_id = j['GBQualifier_value']
print(gene_id)