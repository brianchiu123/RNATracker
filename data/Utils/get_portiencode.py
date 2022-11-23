import requests as rq
from bs4 import BeautifulSoup
import io
import time
from urllib.request import urlopen
import csv
import pandas as pd
import requests

def sleeptime(hour,min,sec):
	return hour*3600 + min*60 + sec;

#######################
#  get uniport link   #
#######################

def get_uniprot_link(gene_id):

	nextlink = "https://www.ncbi.nlm.nih.gov/gene/?term="+gene_id

	nl_response = rq.get(nextlink) 
	if nl_response.status_code != rq.codes.ok:	
		return "null"
	soup = BeautifulSoup(nl_response.text, "lxml") 

	root_1 = soup.findAll('a')

	t = 0
	for url in root_1 :
		#for r in url.findAll('a'):
		if isinstance(url.get('href'), str) :
			if 'www.uniprot.org' in url.get('href'):
				uniprot_link = url.get('href')
				return uniprot_link ;
				t = 1
	if t == 0 :
		return "null" ;

	return uniprot_link ;



#######################
#      get fasta      #
#######################

def get_fasta(uniprot_link):

	fasta_link = uniprot_link + ".fasta?include=yes"
	# print(fasta_link)
	r = rq.get(fasta_link)
	if r.status_code != rq.codes.ok:
		return "null" ;

	response = urlopen(fasta_link)

	fasta = response.read().decode("utf-8", "ignore")

	return fasta ;
	


#######################
#   get all_gene_id   #
#######################
# all_gene_id=[]
# df=pd.read_csv("all-data.csv",encoding= 'unicode_escape')
# df = df.dropna(subset=['Link'])
# gene_id = df[['Gene_ID']]
# f = open("all_gene_id.txt","w")
# for i in range(gene_id.shape[0]):
# 	if gene_id.iloc[i].iat[0].isdigit() :
# 		if gene_id.iloc[i].iat[0] not in all_gene_id:
# 			all_gene_id.append(gene_id.iloc[i].iat[0])
# 			f.write(gene_id.iloc[i].iat[0]+"\n")
#print(all_gene_id)

###########################






# f = open("all_gene_id.txt","r")


f = open('Cytoplasm_indep_gnenid.txt','r')
all_gene_id = f.readlines()
f.close()
fa_out = open("Cytoplasm_indep_protien", "a")
total_out = open("file_0_RNALOCATE.txt",'a')
now=0
count = 1136

total = []
for gene_id in all_gene_id :
	temp = []
	if now <6 :
		now+=1
		print(now)
		continue;
	print(now)
	now+=1
	gene_id=gene_id.strip('\n')
	temp.append(gene_id)
	link = get_uniprot_link(str(gene_id))
	time.sleep(sleeptime(0,0,5))
	temp.append(link)
	temp.append("Cytoplasm")
	s = ""
	for i in temp :
		s=s+i+" "
	total_out.write(s+"\n")
	if link == "null":
		continue ; 

	fasta = get_fasta(link)
	time.sleep(sleeptime(0,0,5))
	if fasta != "null":
		fa_out.write(fasta)
		count+=1
		print("complete "+str(count)+"  "+gene_id)

# print(get_uniprot_link("497652"))

