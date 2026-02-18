import torch
from torch.utils.data import Dataset
from dataset import Vocab
import torchvision.transforms.v2 as T
from PIL import Image
import os
from typing import List
import itertools

class TextTestDataset(Dataset):
    def __init__(
            self,
            file_path,
            de_vocab,
            en_vocab
    ):
        from config import DataConfig
        self.max_length = DataConfig.max_length
        with open(file_path) as f:
            self.de_texts = f.read().splitlines()

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab
    
    def __len__(self):
        return len(self.de_texts)

    def __getitem__(self, item):
        de_text = self.de_texts[item]
        de_encoded = self.de_vocab.encode(de_text)
        de_encoded = [self.bos_id] + de_encoded[:self.max_length-2] + [self.eos_id]
        
        de_length = len(de_encoded)
        de_indices = torch.full((self.max_length,), self.pad_id)
        de_indices[:de_length] = torch.tensor(de_encoded)

        return de_indices, de_length
    

