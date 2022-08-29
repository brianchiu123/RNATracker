PyTorch re-implementation of RNATracker [1] model.

RNATracker aims to predict each RNA sequence belongs to which site in a cell (Cytosol, Nuclear ...)

<img width="422" alt="Screen Shot 2022-08-29 at 11 14 26 AM" src="https://user-images.githubusercontent.com/75982405/187116375-66cb2ef1-6f0b-4a02-80ec-7c36211081e1.png">


## Model Architeture
#### Overview

<img width="420" alt="Screen Shot 2022-08-29 at 11 20 35 AM" src="https://user-images.githubusercontent.com/75982405/187116478-03630088-a73c-4b79-9b19-9138286ebe0e.png">

#### Datail of Each Layer

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

## Reference
[1] Yan Z, Lécuyer E, Blanchette M. Prediction of mRNA subcellular localization using deep recurrent neural networks. Bioinformatics. 2019 Jul 15;35(14):i333-i342. doi: 10.1093/bioinformatics/btz337. PMID: 31510698; PMCID: PMC6612824.
