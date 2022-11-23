import requests as rq
from bs4 import BeautifulSoup
import io
import time
from urllib.request import urlopen
import csv
import pandas as pd
import requests
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import json
import codecs
# ALL_DATA = "Others.fasta"

# Test = "./Training_and_Independent_Dataset/Cytoplasm_indep.fasta"
# f = open(Test, "r")

# w = open('Cytoplasm_indep_gnenid.txt','w')

# gene_id = []
# for line in f :
# 	if line.startswith('>'):
# 		l = line.replace('\n','').split('#')
# 		if l[1] not in gene_id :
# 			gene_id.append(l[1])
# print(gene_id)
# for i in gene_id :
#     # s=""
#     # for j in range(len(i)):
#     #     s=s+i[j]+" "
#     w.write(i+"\n")

service = Service('./chromedriver')
service.start()

def get_XR(transcript_id):

	url = "https://www.ncbi.nlm.nih.gov/nuccore/"+transcript_id
	driver = webdriver.Remote(service.service_url)
	driver.get(url)

	soup = BeautifulSoup(driver.page_source, "lxml") 
	root_1 = soup.find_all('p',{"class":"itemid"})

	for r in root_1 :
		new = r.get_text()
	new_list = new.split(" ")
	for i in new_list :
		if "XR" in i :
			new_link = i
	driver.close()
	driver.quit()
	return new_link+"?report=genbank"


def get_transcripts_seq(transcript_id):

	total_info = []
	url = "https://www.ncbi.nlm.nih.gov/nuccore/"+transcript_id

	# service = Service('./chromedriver')
	# service.start()
	driver = webdriver.Remote(service.service_url)
	driver.get(url)

	time.sleep(10)
	soup = BeautifulSoup(driver.page_source, "lxml") 


	root_1 = soup.find_all('pre')

	for r in root_1:
		info = r.get_text().splitlines()
	try : 	
		for i in info :
			if "VERSION" in i :
				temp = i.replace(" ","")
				version = temp[7:]
	except :
		not_find.write(transcript_id+"\n")
		aa =[transcript_id,"null"]
		driver.close()
		driver.quit()
		return aa
	cds_feature = "feature_"+version+"_CDS_0"
	gene_feature = "feature_"+version+"_gene_0"


	#### cds #####
	root_cds = soup.findAll('span',{'class' : "feature","id":cds_feature})

	for r in root_cds :
		cds_info = r.get_text().split("/") 

	try :
		cds_long_list=cds_info[0].split(";")

		cds_long = cds_long_list[-1].replace(" ","").strip("CDS").strip("\n")

		for i in cds_info : 
			if "translation" in i :
				cds_protien = i.replace(" ","").strip("\n")
				cds_protien = cds_protien[13:-1].replace('\n',"")
	except : 
		cds_long ="null"
		cds_protien="null"

	##### gene long #####
	root_gene = soup.findAll('span',{'class' : "feature","id":gene_feature})

	for r in root_gene :
		gene_info = r.get_text().split("/") 

	try :
		gene_long_list=gene_info[0].split(";")
		gene_long = gene_long_list[-1].replace(" ","").strip("gene").strip("\n")
	except : 
		gene_long = "null"
	##### gene sequence ######

	root_seq = soup.findAll('span',{'class' : "ff_line"})
	gene_seq = ""
	for url in root_seq :
		gene_seq = gene_seq+url.get_text()
	try :
		gene_seq = gene_seq.replace(" ","").strip("\n")
	except :
		gene_seq = "null"
	driver.close()
	driver.quit()

	total_info.append(version)
	total_info.append(gene_long)
	total_info.append(cds_long)
	total_info.append(cds_protien)
	total_info.append(gene_seq)

	return total_info ;
not_find = open("not_find.txt",'a')
data = open("transcript.txt","r")
a = open("file0_ALL_DATA.txt","a")
now = 0 
done = 11376
for line in data :
	line=line.strip('\n')
	if now < done :
		now+=1
		continue;

	line_list = line.split(" ")
	if "_" not in line_list[2] :
		continue ;
	transcript_id = line_list[2].strip("\n")

	if "XR" in transcript_id :
		transcript_id=get_XR(transcript_id)
	now_info = get_transcripts_seq(transcript_id)

	s=""
	for i in now_info : 
		s = s + i + " "
	a.write(s+"\n")
	time.sleep(5)
	now+=1
	print(str(now)+" "+transcript_id)




