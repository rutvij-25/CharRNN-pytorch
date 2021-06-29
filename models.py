import torch.nn as nn
import torch

class CharRNN(nn.Module):

    def __init__(self,vocab_size,embed_dim,hidden_size,num_layers,p):

        super(CharRNN,self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        self.LSTM = nn.LSTM(input_size=embed_dim,hidden_size = hidden_size,num_layers=num_layers ,batch_first=True,dropout = p)
        self.linear = nn.Linear(hidden_size,vocab_size)

    def forward(self,x,h,c):

        embed = self.embedding(x)
        output,(hidden,cell) = self.LSTM(embed,(h,c))
        output = output.reshape(output.shape[0]*output.shape[1],self.hidden_size)
        lin_out = self.linear(output)
        return lin_out,(hidden,cell)

    def init_hidden(self,batch_size):
        return torch.zeros((self.num_layers,batch_size,self.hidden_size))

    def init_cell(self,batch_size):
        return torch.zeros((self.num_layers,batch_size,self.hidden_size))
