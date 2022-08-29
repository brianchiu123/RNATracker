import torch
from torch import nn
from torch.nn.modules import dropout

class Attention(nn.Module):
    def __init__(self, input_size,
                        attention_size = 50):
        super().__init__()
        self.attention_size = attention_size
        self.input_size = input_size

        self.attention_w = nn.Linear(self.input_size, 
                                    self.attention_size,
                                    bias = True)
        self.tanh = nn.Tanh()
        self.attention_w2 = nn.Linear(self.attention_size, 
                                    1,
                                    bias = False)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, input):
        # input : [length, batch, features]
        
        # calculate attention score
        attn = self.attention_w(input)
        attn = self.tanh(attn)                                      #[length, batch, attention_size]
        attn = self.attention_w2(attn)                              #[length, batch , 1]
        attn = torch.transpose(torch.squeeze(attn), 0, 1)           #[length, batch]

        # softmax
        alphas = self.softmax(attn)
        alphas = torch.Tensor.reshape(alphas, [-1, len(input), 1])  #[length, batch, 1]

        # multiply attention score to original rnn output and sum over seq length
        rnn_output = torch.transpose(input, 0, 1)                   #[length, batch, features]
        attn_output = torch.sum(rnn_output * alphas, 1)             #[batch, features]

        return attn_output


class RNATracker_model(nn.Module):
    def __init__(self, input_size, 
                        output_size,
                        device):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rnn_hidden_size = 32
        self.bidirectional = True
        self.layer_size = 2 if self.bidirectional else 1

        
        # define layers
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels = self.input_size, 
                              out_channels = 32, 
                              kernel_size = 10, 
                              stride = 1,
                              bias = False),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.25)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels = 32, 
                              out_channels = 32, 
                              kernel_size = 10, 
                              stride = 1,
                              bias = False),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=3),
            nn.Dropout(0.25)
        )
        
        self.gru = nn.GRU(32, 
                          self.rnn_hidden_size, 
                          num_layers = 1, 
                          batch_first = True, 
                          bidirectional = self.bidirectional)

        self.attention = Attention(attention_size = 50,
                                    input_size = self.rnn_hidden_size * self.layer_size)
        self.acti = nn.ReLU()

        self.output = nn.Linear(self.rnn_hidden_size * self.layer_size, self.output_size)
        
        self.softmax = nn.Softmax(dim=1)

        # weight initialization
        nn.init.xavier_uniform_(self.conv_block1[0].weight)
        nn.init.xavier_uniform_(self.conv_block2[0].weight)


    def forward(self, input):
        input = torch.transpose(input, 1, 2)     # conv input [batch, channel, length]
        conv_out = self.conv_block1(input)
        conv_out = self.conv_block2(conv_out)

        conv_out = torch.transpose(conv_out, 1, 2)    # rnn input [batch, length, feature]
        gru_out, h_n = self.gru(conv_out)
        gru_out = self.acti(gru_out)

        gru_out = torch.transpose(gru_out, 0, 1)    # attention input [length, batch, feature]
        attn_output = self.attention(gru_out)
        attn_output = self.acti(attn_output)

        out = self.output(attn_output)
        out = self.softmax(out)
        return out