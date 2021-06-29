import torch
from models import *

checkpoint = torch.load('pretrained/saved_model.pth',map_location=torch.device('cpu'))

model = checkpoint['model']

model.load_state_dict(checkpoint['model_state_dict'])