import numpy as np 
import torch
from torch.utils.data import Dataset
    
    
def load_cefra(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    ids = []
    seqs = []
    labels = []
    flag = 0
    for line in lines:
        line = line.split('\n')[0]
        flag += 1
        if flag == 1:
            ids.append(line.split(' ')[-1])
        elif flag == 2 :
            seqs.append(line)
        elif flag == 3 :
            line = line.split(' ')
            labels.append(np.array(line, dtype = float))
            flag = 0
    return {'seqs' : np.array(seqs), 'labels' : np.array(labels)}


def onehot_dna(seq):
    # parse sequence in GATC order 
    onehot_seq = []
    for i in seq:
        if i == 'G':
            onehot_seq.append([1,0,0,0])
        elif i == 'A':
            onehot_seq.append([0,1,0,0])
        elif i == 'T':
            onehot_seq.append([0,0,1,0])
        elif i == 'C':
            onehot_seq.append([0,0,0,1])
        elif i == '_':
            onehot_seq.append([0,0,0,0])  # padding sign
        else:
            raise ValueError(f"undefined character \'{i}\'")
    
    if len(seq) == 0: onehot_seq.append([0, 0, 0, 0])
    return np.array(onehot_seq)


def norm_label(label):
    sumup = np.sum(label)
    return np.array([round(i/sumup, 4) for i in label])


# pad and trim sequence on the front
def trim_and_pad_seq(seq, length = 4000, pad_sign = "_"):
    if len(seq) > length:
        seq = seq[:length]
    elif len(seq) == length:
        pass
    else:
        pad_length = length - len(seq)
        for i in range(pad_length):
            seq = pad_sign + seq
    
    return seq

# pad and trim the one-hot matrix on the front
def trim_and_pad(seq, length = 4000):
    seq = np.array(seq)
    if len(seq) > length:
        seq = seq[:length]
    elif len(seq) == length:
        pass
    else:
        pad_length = length - len(seq)
        pad_array = np.zeros(shape = (pad_length, seq.shape[-1]), dtype = float)
        seq = np.concatenate((pad_array, seq), axis=0)
    
    return seq


# for batch manipulation
def pad_collate(batch):
    X = [item[0] for item in batch]
    Y = [item[1] for item in batch]

    # this process is transfrer in model forward pass
    # and no need in RNATracker
    """
    # sort sequences by length (must)
    X.sort(key = lambda x: len(x), reverse = True) 

    # get sequence lengths
    lens = [len(i) for i in X]
    """

    """
    # pad sequence (output is torch.tensor)
    X = [torch.from_numpy(x).type(torch.float) for x in X]
    X = pad_sequence(X, batch_first = True)
    """
    X = torch.tensor(np.array(X)).type(torch.float)
    Y = torch.tensor(np.array(Y)).type(torch.float)
    
    return [X, Y]


# can only manipulate indivisually
class seqDataset(Dataset):
    def __init__(self, X, y, loc, one_hot = False, max_length = 4000, full_length = False):
        self.n_samples = len(y)
        self.x_data = X
        self.y_data = y
        self.loc = loc
        self.one_hot = one_hot
        self.max_length = max_length
        self.full_length = full_length

    def __getitem__(self, index):
        return_x = self.x_data[index]
        return_y = norm_label(self.y_data[index])
        if self.one_hot:
            return_x = onehot_dna(return_x)
        if not self.full_length:
            return_x = trim_and_pad(return_x, self.max_length)

        return return_x, return_y

    def __len__(self):
        return self.n_samples