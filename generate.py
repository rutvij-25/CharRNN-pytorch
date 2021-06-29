import torch
import torch.nn as nn
import argparse
from dataset import *
from models import *
argparser = argparse.ArgumentParser()

argparser.add_argument('--initialtext',type=str,default="Where")
argparser.add_argument('--temperature',type=int,default=1)
argparser.add_argument('--n',type=int,default=100)
argparser.add_argument('--root',type=str,default='data.txt')

args = argparser.parse_args()

def sample(model,data,n,initial_text,device,temperature=1.0):
  encoded = torch.LongTensor(data.encode(initial_text)).unsqueeze(0).to(device)
  output_s = []
  model.eval()
  softmax = nn.Softmax(dim=0)
  with torch.no_grad():
    h1 = model.init_hidden(1).to(device)
    c1 = model.init_cell(1).to(device)
    for i in range(n):
      output,(h1,c1) = model(encoded,h1,c1)
      output = torch.log(softmax(output[-1]))/temperature
      exp_preds = torch.exp(output)
      preds = exp_preds / torch.sum(exp_preds)
      probas = torch.multinomial(preds,1)[0].item()
      output_s.append(probas)
      encoded = torch.LongTensor([probas]).unsqueeze(1).to(device)
    
  return initial_text + "".join(data.decode(output_s))





device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load('pretrained/saved_model.pth',map_location=device)
model = checkpoint['model']
model.load_state_dict(checkpoint['model_state_dict'])
dataset = TextData(args.root,100)
    
print(sample(model,dataset,args.n,args.initialtext,device,args.temperature))