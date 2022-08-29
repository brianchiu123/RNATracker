import configparser
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

from tqdm import tqdm

from utils.data import pad_collate, seqDataset, load_cefra
from models.model import RNATracker_model
from utils.metric import pearson_correlation_by_class

def test(opt, config, device, test_data, in_training = False):

    # dataset and dataloader
    loc = config['model']['loc_name'].split(',')
    test_dataset = seqDataset(test_data['seqs'], test_data['labels'], loc, 
                                one_hot = True,
                                max_length = opt.max_length,
                                full_length = opt.full_length)
    test_loader = DataLoader(dataset = test_dataset,
                            batch_size = opt.batch_size,
                            shuffle = False,
                            collate_fn = pad_collate)

    # model definition
    num_classes = config['model'].getint('num_classes')
    input_channel = 4  # one-hot of ATCG
    model = RNATracker_model(input_channel = input_channel, output_size = num_classes,).to(device)
    criterion = nn.KLDivLoss(reduction='sum')

    # load weights
    ckpt = torch.load(opt.weights, map_location = device)
    model.load_state_dict(ckpt['model'])

    # test 
    test_loss = 0
    test_true_label = []
    test_pred_label = []
    pbar = tqdm(enumerate(test_loader), total = len(test_loader))
    with torch.no_grad():
        for i, (inputs, labels) in pbar:
            labels = labels.to(device)
            
            # forward
            outputs = model(inputs)

            # calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # collect labels
            test_true_label.append(labels.cpu().numpy())
            test_pred_label.append(outputs.cpu().numpy())
    
    # metric calculation
    test_true_label = np.concatenate(test_true_label, axis = 0)
    test_pred_label = np.concatenate(test_pred_label, axis = 0)
    pcc_dict = pearson_correlation_by_class(test_true_label, test_pred_label, loc)
    
    # write test descriptopn 
    test_desc = f'Test Loss={test_loss/len(test_loader):.4f}\nPCC:{pcc_dict}'
    print(test_desc)


if __name__ == '__main__':
    # load config file
    config = configparser.ConfigParser()
    config.read('cfg/settings.cfg')

    # parse script options
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='load weights path')
    parser.add_argument('--test_data', type=str, required=True, help='testing data path')
    parser.add_argument('--max_length', type=int, default = 4000, help='pad and trim seq to a fixed length')
    parser.add_argument('--batch_size', type=int, default=config['hyperparameters'].getint('batch_size'), help='total batch size')
    opt = parser.parse_args()

    # device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'using {device} : {torch.cuda.get_device_name()}\n')
    else:
        print(f'using {device}\n')
    
    # load test data
    test_seqs, test_labels = load_cefra(opt.test_data)
    test_data = dict({'seqs': test_seqs, 'labels' : test_labels})
    print(f'[ Using testing data in \"{opt.test_data}\"]')
    print(f'Testing size : ', {len(test_data['seqs'])},'\n')

    test(opt, config, device, test_data)