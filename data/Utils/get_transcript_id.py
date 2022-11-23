import csv
import pandas as pd

ALL_DATA = "Others.fasta"
f = open(ALL_DATA , "r")


w = open("transcript.txt","w")
for line in f :

	if line.startswith('>'):
		info = line.split("|")
		gene_id = info[3]
		t = info[5].split(" ")
		t_id = t[0].replace("\n","").strip('\n')

		gene_name = t[1]
		s = gene_id + " "+ gene_name+ " " + t_id +"\n"
		w.write(s)
