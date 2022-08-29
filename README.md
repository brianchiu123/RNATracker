### A PyTorch re-implementation of RNATracker [1] model (non-official).


#### Brief Intro of RNATracker
RNATracker aims to predict each RNA sequence belongs to which site in a cell (Cytosol, Nuclear ...). It utilize a combination of CNN, RNN model with Attention Mechanism. Target label is not hard label, it's a vector represent the probability of being exist in a specific site. Thus, it's more like a regression task and the metric used here is Pearson Correlation Coefficient.

<img width="422" alt="Screen Shot 2022-08-29 at 11 14 26 AM" src="https://user-images.githubusercontent.com/75982405/187116375-66cb2ef1-6f0b-4a02-80ec-7c36211081e1.png">


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
    (0): Conv1d(4, 32, kernel_size=(10,), stride=(1,), bias=False)
    (1): ReLU()
    (2): MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.25, inplace=False)
  )
  (conv_block2): Sequential(
    (0): Conv1d(32, 32, kernel_size=(10,), stride=(1,), bias=False)
    (1): ReLU()
    (2): MaxPool1d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
    (3): Dropout(p=0.25, inplace=False)
  )
  (gru): GRU(32, 32, batch_first=True, bidirectional=True)
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
#### Pearson Correlation Coefficient of each site

#### Training / Validation loss 
![螢幕快照 2022-08-29 下午2 28 12](https://user-images.githubusercontent.com/75982405/187137091-417bae00-49f0-4319-ab23-803d4ae7a581.png)

![螢幕快照 2022-08-29 下午2 28 20](https://user-images.githubusercontent.com/75982405/187137139-a8ac06b4-ff5c-4d00-8a33-84d0898289d5.png)![螢幕快照 2022-08-29 下午2 28 33](https://user-images.githubusercontent.com/75982405/187137151-0cfe4d97-e375-42d0-bd02-3613e1ec093c.png)


<br>

## Reference
[1] Yan Z, Lécuyer E, Blanchette M. Prediction of mRNA subcellular localization using deep recurrent neural networks. Bioinformatics. 2019 Jul 15;35(14):i333-i342. doi: 10.1093/bioinformatics/btz337. PMID: 31510698; PMCID: PMC6612824.
