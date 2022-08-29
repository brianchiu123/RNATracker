import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../data/Cefra/cefra_95.fa', help='input fasta file')
parser.add_argument('--train_ratio', type=float, default=0.9, help='training set ratio, range [0,1]')
opt = parser.parse_args()

# read input fasta
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
print('all : ', len(ids))

# calculate train and test id
train_num = int(len(ids) * opt.train_ratio)
train_id = np.random.choice(range(0, len(ids)), size = train_num, replace=False)
print('train number : ', len(train_id))
test_id = np.array([i for i in range(0, len(ids), 1) if not i in train_id])
print('test number: ', len(test_id))

# write
train_path = opt.input.split('.fa')[0] + '_train.fa'
with open(train_path, 'w+') as f:
    for id in train_id:
        f.write('> ')
        f.write(ids[id])
        f.write('\n')
        f.write(proteins[id])
        f.write('\n')
        f.write(five_utrs[id])
        f.write('\n')
        f.write(cdss[id])
        f.write('\n')
        f.write(three_utrs[id])
        f.write('\n')
        f.write(' '.join(labels[id]))
        f.write('\n')

test_path = opt.input.split('.fa')[0] + '_test.fa'
with open(test_path, 'w+') as f:
    for id in test_id:
        f.write('> ')
        f.write(ids[id])
        f.write('\n')
        f.write(proteins[id])
        f.write('\n')
        f.write(five_utrs[id])
        f.write('\n')
        f.write(cdss[id])
        f.write('\n')
        f.write(three_utrs[id])
        f.write('\n')
        f.write(' '.join(labels[id]))
        f.write('\n')
