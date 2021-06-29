import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optimizers
from torch.utils.data import DataLoader
import argparse
from dataset import TextData
from models import *
from generate import *

argparser = argparse.ArgumentParser()

argparser.add_argument('--embed_dim',type=int,default=65)
argparser.add_argument('--dropout',type=float,default=0.1)
argparser.add_argument('--hidden_size',type=int,default=50)
argparser.add_argument('--num_layers',type=int,default=2)
argparser.add_argument('--batch_size',type=int,default=100)
argparser.add_argument('--chunk_size',type=int,default=200)
argparser.add_argument('--epochs',type=int,default=200)
argparser.add_argument('--lr',type=float,default=0.01)
argparser.add_argument('--root',type=str,default='data.txt')

args = argparser.parse_args()

dataset = TextData(args.root,args.chunk_size)
dataloader = DataLoader(dataset,args.batch_size,shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab_size = len(dataset)

model = CharRNN(vocab_size,args.embed_dim,args.hidden_size,args.num_layers,args.dropout).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optimizers.Adam(model.parameters(),args.lr)

for epoch in range(args.epochs):
  
    for input,target in dataloader:
      
        model.train()
        batch_size = input.shape[0]
        input = input.to(device)
        target = target.to(device)
        h = model.init_hidden(batch_size)
        c = model.init_cell(batch_size)
        h = h.to(device)
        c = c.to(device)
        output,(_,_) = model(input,h,c)
        loss = criterion(output,target.view(-1))
        model.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch%50 == 0:
        checkpoint = {
          
          'model_state_dict':model.state_dict(),
          'opt_state_dict':optimizer.state_dict()
      }
        torch.save(checkpoint,f'pretrained/model{epoch}.pt')

    if epoch%10 == 0:
        print(f'EPOCH:{epoch} LOSS:{loss}')
        print(sample(model,dataset,50,"Where"))

