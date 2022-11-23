### A PyTorch re-implementation of RNATracker [1] model (non-official).


#### Brief Intro of RNATracker
RNATracker aims to predict each RNA sequence belongs to which site in a cell (Cytosol, Nuclear ...). It utilize a combination of CNN, RNN model with Attention Mechanism (self-attention). Target label is not hard label, it's a vector represent the probability of being exist in a specific site. Thus, it's more like a regression task and the metric used here is Pearson Correlation Coefficient.

<img width="422" alt="Screen Shot 2022-08-29 at 11 14 26 AM" src="https://user-images.githubusercontent.com/75982405/187116375-66cb2ef1-6f0b-4a02-80ec-7c36211081e1.png">

<br>

## Data Collection

#### Data Sources
We collected data from four different data sets, compared and pre-processed these data to obtain the data we need and available.

<img width="647" alt="截圖 2022-11-23 下午8 58 50" src="https://user-images.githubusercontent.com/29274119/203554136-43b7f7a0-f05c-43ae-b47e-6807fc523e24.png">



<br>

#### Data Processing

Using NCBI API and Ensembl Biomart can convert labels and obtain the data we need, such as protein sequences, and can also divide gene sequences into CDS, 3'UTR and 5'UTR.Then compare these data and do some sequence similarity comparisons to find the sequence we need.In this project we will compare with the Apex dataset and Cefra dataset in RNATracker.


<img width="408" alt="截圖 2022-11-23 下午9 25 38" src="https://user-images.githubusercontent.com/29274119/203558208-dcb9d2be-513a-42b0-b9fa-03a1316c958c.png">

<img width="519" alt="截圖 2022-11-23 下午9 25 44" src="https://user-images.githubusercontent.com/29274119/203558319-d9486c75-c498-4c0a-83c9-2d47727a3158.png">




<br>

## Model Architeture
#### Overview



|Overall Model                  |Illustartion of Attention Layer              |
|-------------------------------|-----------------------------|
| <img width="420" alt="Screen Shot 2022-08-29 at 11 20 35 AM" src="https://user-images.githubusercontent.com/75982405/187116478-03630088-a73c-4b79-9b19-9138286ebe0e.png">|<img width="312" alt="Screen Shot 2022-08-29 at 11 47 25 AM" src="https://user-images.githubusercontent.com/75982405/187119145-c18bea65-8094-4a6c-9397-4e26b6eeb8ba.png">|


#### Datail of Each Layer (actual implemation of this repo)

```
RNATracker_model(
  (conv_block1): Sequential(
    (0): Conv1d(in_channels=4, out_channels=32, kernel_size=(10,), stride=(1,), bias=False)
    (1): ReLU()
    (2): MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.25, inplace=False)
  )
  (conv_block2): Sequential(
    (0): Conv1d(in_channels=32, out_channels=32, kernel_size=(10,), stride=(1,), bias=False)
    (1): ReLU()
    (2): MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.25, inplace=False)
  )
  (gru): GRU(input_size=32, hidden_size=32, batch_first=True, bidirectional=True)
  (attention): Attention(
    (attention_w): Linear(in_features=64, out_features=50, bias=True)
    (tanh): Tanh()
    (attention_w2): Linear(in_features=50, out_features=1, bias=False)
    (softmax): Softmax(dim=-1)
  )
  (acti): ReLU()
  (output): Linear(in_features=64, out_features=4, bias=True)
  (softmax): Softmax(dim=1)
)
```



<br>

## Usage
``` 
usage: train.py [-h] [--data DATA] [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR] [--max_length MAX_LENGTH] [--log_name LOG_NAME]
                [--weights WEIGHTS] [--fold FOLD] [--full_length]

optional arguments:
  -h, --help                show this help message and exit
  --data DATA               RNA data path
  --epochs EPOCHS           num of epochs
  --batch_size BATCH_SIZE   total batch size
  --lr LR                   learning rate
  --max_length MAX_LENGTH   pad and trim seq to a fixed length
  --log_name LOG_NAME       name for this run
  --weights WEIGHTS         load weights path
  --fold FOLD               fold number(do not set if do not want), will use training data to run n-fold
  --full_length             use full length to train (batch_size will set to 1)
```
- default args of `epochs`, `batch_size`, `max_length` ... are the one used in original RNATracker.
- every training will create a log folder under `logs`, will named `exp<number>`, or other log name if `log_name` option is given.
- if you want to perform full length training (without padding and trimmimg to a fixed size `max_length`), you can set `full_length`, and the `batch_size` will automatically set to 1. 
- you can restore and keep on last training by giving `weights`, epoch will start from the last time.
- you can perform k-fold cross validation by setting `fold` option.

<br>

#### config file

in `config/setting.cfg`, there's some conventional option can be set. Include all log file name, folder name...

```
[model]
num_classes = 4
loc_name = Cytosol,Insoluble,Membrane,Nuclear

[log]
save_weights_per_epoch = 5
eval_per_epoch = 5
overall_log_dir = logs
log_dir_prefix = exp
log_file_name = log.txt

```


<br>

## Result (of Cefra Dataset)

training log is in `logs/exp0`, recorded in both plain text and tensorboard.

#### Pearson Correlation Coefficient of each site
|Site|PCC|
|----|---|
|Cytosol|0.7119|
|Insoluble|0.6356|
|Membrane|0.5675|
|Nuclear|0.5575|
|overall|0.6333|

#### Training / Validation loss 
![螢幕快照 2022-08-29 下午2 28 12](https://user-images.githubusercontent.com/75982405/187137091-417bae00-49f0-4319-ab23-803d4ae7a581.png)

![螢幕快照 2022-08-29 下午2 28 20](https://user-images.githubusercontent.com/75982405/187137139-a8ac06b4-ff5c-4d00-8a33-84d0898289d5.png)![螢幕快照 2022-08-29 下午2 28 33](https://user-images.githubusercontent.com/75982405/187137151-0cfe4d97-e375-42d0-bd02-3613e1ec093c.png)


<br>

## Reference
[1] Yan Z, Lécuyer E, Blanchette M. Prediction of mRNA subcellular localization using deep recurrent neural networks. Bioinformatics. 2019 Jul 15;35(14):i333-i342. doi: 10.1093/bioinformatics/btz337. PMID: 31510698; PMCID: PMC6612824.  
[2]RNALocate: a resource for RNA subcellular localizations. Nucleic Acids Res. 2017 Jan 4,Zhang T, Tan P, Wang L, Jin N, Li Y, Zhang L, Yang H, Hu Z, Zhang L, Hu C, Li C, Qian K, Zhang C, Huang Y, Li K, Lin H, Wang D.  
[3]Live-cell mapping of organelle-associated RNAs via proximity biotinylation combined with protein-RNA crosslinking, eLife 2017;6, Pornchai Kaewsapsak, David Michael Shechner, William Mallard, John L Rinn, Alice Y Ting.  
[4]CeFra-seq reveals broad asymmetric mRNA and noncoding RNA distribution profiles in Drosophila and human cells. RNA. 2018 Jan ,Benoit Bouvrette LP, Cody NAL, Bergalet J, Lefebvre FA, Diot C, Wang X, Blanchette M, Lécuyer E.  
[5]Atlas of Subcellular RNA Localization Revealed by APEX-Seq,Cell,Volume 178, Issue 2,2019 ,Furqan M. Fazal, Shuo Han, Kevin R. Parker, Pornchai Kaewsapsak, Jin Xu, Alistair N. Boettiger, Howard Y. Chang, Alice Y. Ting.  
