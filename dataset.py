from torch.utils.data import Dataset
import torch

class TextData(Dataset):

    def __init__(self,root,chunk_size):
        self.root = root
        self.chunk_size = chunk_size
        with open(root) as f:
            self.lines = f.read()
        self.char2idx = {i:j for (j,i) in enumerate(sorted(set(self.lines)))}
        self.idx2char = {i:j for (i,j) in enumerate(sorted(set(self.lines)))}
    
    def encode(self,x):
        return [self.char2idx[i] for i in x]
    
    def decode(self,x):
        return [self.idx2char[i] for i in x]
    
    def chunk(self,x):
        output=[x[i:i + self.chunk_size] for i in range(0, len(x), self.chunk_size)]
        return output
    
    def __len__(self):
        return len(self.lines)//self.chunk_size
    
    def __getitem__(self, index):
        X = self.split_input_target(self.encode(self.lines))[0]
        y = self.split_input_target(self.encode(self.lines))[1]
        X = torch.LongTensor(self.chunk(X)[:-1])
        y = torch.LongTensor(self.chunk(y)[:-1])
        return X[index],y[index]

    @staticmethod
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return (input_text, target_text)