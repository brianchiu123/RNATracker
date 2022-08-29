import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../data/Cefra/cefra_95_train.fa', help='input fasta file')
parser.add_argument('--output_dir', type=str, default='../data/Cefra/protein_seq_train', help='output fasta directory')
opt = parser.parse_args()

# read .fa
with open(opt.input, 'r') as f:
    lines = f.readlines()
    ids = []
    proteins = []
    five_utrs = []
    cdss = []
    three_utrs = []
    labels = []
    flag = 0
    for line in lines:
        line = line.split('\n')[0]
        flag += 1
        if flag == 1:
            ids.append(line.split(' ')[-1])
        elif flag == 2 :
            proteins.append(line)
        elif flag == 3 :
            five_utrs.append(line)
        elif flag == 4 :
            cdss.append(line)
        elif flag == 5 :
            three_utrs.append(line)
        elif flag == 6 :
            line = line.split(' ')
            labels.append(np.array(line, dtype = str))
            flag = 0

# check output dir
if not os.path.isdir(opt.output_dir):
    os.mkdir(opt.output_dir)

# output every sequence as a fasta format
file_dir = opt.output_dir + '/'
for i, id in enumerate(ids):
    file_name = id+'.fasta'
    with open(file_dir + file_name, 'w+') as f:
        f.write('>')
        f.write(id)
        f.write('\n')
        f.write(proteins[i])
        f.write('\n')
