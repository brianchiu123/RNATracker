import configparser
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.data import load_cefra, pad_collate, seqDataset
from utils.run import increment_dir
from models.model import RNATracker_model
from utils.metric import pearson_correlation_by_class

def train(opt, config, device, train_data, val_data):

    # run log dir setting
    runs_dir, log_file_name = config['log']['runs_dir'], config['log']['log_file_name']
    if not os.path.isdir(runs_dir): os.mkdir(runs_dir)
    run_dir = increment_dir(runs_dir, opt.run_name)  #dir name for this run
    os.mkdir(run_dir)
    weights_dir = run_dir + '/weights'
    os.mkdir(weights_dir)
    log_file_path = run_dir +  '/' + log_file_name
    tb_writer = SummaryWriter(log_dir = run_dir)

    # write data info
    train_size = len(train_data['seqs'])
    train_data_desc = f'[ Using training data in \"{opt.train_data}\"] \nTraining size : {train_size}'
    val_size = len(val_data['seqs'])
    val_data_desc = f'[ Using validataion data in \"{opt.val_data}\"] \nValidataion size : {val_size}\n'
    length_desc = f'trim and pad to length {opt.max_length}\n'
    print(train_data_desc)
    print(val_data_desc)
    print(length_desc)
    with open(log_file_path, 'a+') as f:
        f.write(train_data_desc)
        f.write('\n')
        f.write(val_data_desc)
        f.write('\n')
        f.write(length_desc)
        f.write('\n')

    # full length mode
    if opt.full_length:
        opt.batch_size = 1
    
    # dataset and dataloader
    loc = config['model']['loc'].split(',')
    max_length = opt.max_length
    train_dataset = seqDataset(train_data['seqs'], train_data['labels'], loc, 
                                one_hot = True, 
                                max_length = max_length,
                                full_length = opt.full_length)
    val_dataset = seqDataset(val_data['seqs'], val_data['labels'], loc, 
                                one_hot = True,
                                max_length = max_length,
                                full_length = opt.full_length)
    train_loader = DataLoader(dataset = train_dataset,
                          batch_size = opt.batch_size,
                          shuffle = True,
                          collate_fn = pad_collate)
    val_loader = DataLoader(dataset = val_dataset,
                            batch_size = opt.batch_size,
                            shuffle = False,
                            collate_fn = pad_collate)

    # model definition
    num_classes = config['model'].getint('num_classes')
    input_channel = 4  # one-hot of ATCG
    model = RNATracker_model(input_channel = input_channel, output_size = num_classes,).to(device)
    criterion = nn.KLDivLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    start_epoch = 1

    # load weights and optimizar
    if opt.weights:
        ckpt = torch.load(opt.weights, map_location = device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        load_desc = f'[ Loaded model and optimizer stat]\nstart epoch : {start_epoch}\n'
        print(load_desc)
        with open(log_file_path, 'a+') as f:
            f.write(load_desc)
            f.write('\n')


    # output training info
    info_desc = f'[ Training information ]\noutput file directory : {run_dir}, batch size : {opt.batch_size}, learning rate : {opt.lr}, total epochs : {opt.epochs}\n{model}\n{optimizer}\n'
    print(info_desc)
    with open(log_file_path, 'a+') as f:
        f.write(info_desc)


    # training loop
    eval_per_epoch, save_weights_per_epoch = config['log'].getint('eval_per_epoch'), config['log'].getint('save_weights_per_epoch')
    for epoch in range(start_epoch, opt.epochs+1):
        train_loss = 0
        pbar = tqdm(enumerate(train_loader), total = len(train_loader))

        # iteration
        for i, (inputs, labels) in pbar:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(inputs)
            
            # calculate loss
            log_outputs = torch.log(outputs)
            loss = criterion(log_outputs, labels)
            train_loss += loss.item()
    
            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # update train description
            train_desc = f'Epoch [{epoch}/{opt.epochs}], Loss: {train_loss/(i+1):.4f}'
            pbar.set_description(desc = train_desc)


        # validation
        if epoch % eval_per_epoch == 0:
            val_loss = 0
            val_truth_labels = []
            val_pred_labels = []
            with torch.no_grad():
                for (inputs, labels) in val_loader:
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # forward
                    outputs = model(inputs)

                    # calculate loss
                    log_outputs = torch.log(outputs)
                    loss = criterion(log_outputs, labels)
                    val_loss += loss.item()
                    
                    # collect labels
                    val_truth_labels.append(labels.cpu().numpy())
                    val_pred_labels.append(outputs.cpu().numpy())
            
            # calculate metric
            val_truth_labels = np.concatenate(val_truth_labels, axis = 0)
            val_pred_labels = np.concatenate(val_pred_labels, axis = 0)
            pcc_dict = pearson_correlation_by_class(val_truth_labels, val_pred_labels, loc)
            
            # write val descriptopn (after val end)
            val_desc = f'Val Loss={val_loss/len(val_loader):.4f}\nPCC:{pcc_dict}'
            print(val_desc)


        # write log file
        with open(log_file_path, 'a+') as f:
            f.write(train_desc)
            f.write('\n')
            if epoch % eval_per_epoch == 0 :
                f.write(val_desc)
                f.write('\n')
            

        # write tensorboard
        tb_writer.add_scalar('Train/Loss', train_loss/len(train_loader), epoch)
        if epoch % eval_per_epoch == 0:
            tb_writer.add_scalar('Val/Loss', val_loss/len(val_loader), epoch)
            tb_writer.add_scalar('Val/overall_pcc', pcc_dict['overall'], epoch)

        
        # save weights
        if epoch % save_weights_per_epoch == 0:
            weights_path = weights_dir + '/' + 'epoch' + str(epoch) + '.pt'
            ckpt = {'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
            torch.save(ckpt, weights_path)

    tb_writer.close()


if __name__ == '__main__':
    # load config file
    config = configparser.ConfigParser()
    config.read('cfg/settings.cfg')

    # parse script options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/Cefra/cefra.fa', help='RNA data path')
    parser.add_argument('--epochs', type=int, default=100, help='num of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='total batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--max_length', type=int, default = 4000, help='pad and trim seq to a fixed length')
    parser.add_argument('--run_name', type=str, help='name for this run')
    parser.add_argument('--weights', type=str, help='load weights path')
    parser.add_argument('--fold', type=int, help='fold number(do not set if do not want), will use training data to run n-fold')
    parser.add_argument('--full_length', action="store_true", help='use full length to train (batch_size will set to 1)')
    opt = parser.parse_args()

    # device
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'using {device} : {torch.cuda.get_device_name()}\n')
    else:
        print(f'using {device}\n')
    
    # load data
    if opt.fold:
        whole_data = load_cefra(opt.data)

        kf = KFold(n_splits = opt.fold)
        fold = 0
        base_name = opt.run_name
        for train_idx , val_idx in kf.split(whole_data['seqs']):
            train_idx , val_idx= list(train_idx), list(val_idx)
            fold  = fold + 1
            opt.run_name = f'{base_name}_fold{fold}'
            fold_train_data = {'seqs' : [whole_data['seqs'][i] for i in train_idx], 'labels' : [whole_data['labels'][i] for i in train_idx]}
            fold_val_data = {'seqs' : [whole_data['seqs'][i] for i in val_idx], 'labels' : [whole_data['labels'][i] for i in val_idx]}
            train(opt, config, device, fold_train_data, fold_val_data)
    else:
        whole_data = load_cefra(opt.data)
        train_seqs, train_labels, val_seqs, val_labels = train_test_split(whole_data['seqs'],whole_data['labels'],test_size=0.2, random_state=1)
        train(opt, config, device, {"seqs" : train_seqs, "labels" : train_labels}, {"seqs" : val_seqs, "labels" : val_labels})