import torch
from torch.utils.data import Dataset

class TextData(Dataset):
    
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
        
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, i):
        x = self.tokens[i : i + self.block_size]
        y = self.tokens[i + 1 : i + self.block_size + 1]
        
        return x, y
    
class TextFile:
    
    def __init__(self,
                 file_path,
                 split_ratio=0.9):
        
        with  open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = { i:ch for i,ch in enumerate(self.vocab)}
        self.encode = lambda s : [self.stoi[c] for c in s]
        self.decode = lambda e : ''.join([self.itos[i] for i in e])

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(split_ratio*(len(data)))
        
        self.train = data[:n] 
        self.test = data[n:]
        
    def tokens(self, split='train'):
        tokens = getattr(self, split)
        return tokens