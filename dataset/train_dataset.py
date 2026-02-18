import torch
from torch.utils.data import Dataset
from dataset import Vocab
import torchvision.transforms.v2 as T
from PIL import Image
import os
from typing import List
import itertools

class TextTrainDataset(Dataset):
    def __init__(
            self,
            texts: List[str],
            de_vocab=None,
            en_vocab=None
    ):
        from config import DataConfig
        self.max_length = DataConfig.max_length
        self.de_texts = [el[0] for el in texts]
        self.en_texts = [el[1] for el in texts] # List[str]

        self.pad_id, self.unk_id, self.bos_id, self.eos_id = 0, 1, 2, 3
        if not de_vocab:
            self.de_vocab = Vocab(prefix='de')
            self.en_vocab = Vocab(prefix='en')
        else:
            self.de_vocab = de_vocab
            self.en_vocab = en_vocab
    
    def __len__(self):
        return len(self.de_texts)

    def __getitem__(self, item):
        de_text = self.de_texts[item]
        en_text = self.en_texts[item]

        de_encoded = self.de_vocab.encode(de_text)
        de_encoded = [self.bos_id] + de_encoded[:self.max_length-2] + [self.eos_id]
        en_encoded = self.en_vocab.encode(en_text)
        en_encoded = [self.bos_id] + en_encoded[:self.max_length-2] + [self.eos_id]
        
        de_length = len(de_encoded)
        de_indices = torch.full((self.max_length,), self.pad_id)
        de_indices[:de_length] = torch.tensor(de_encoded)
        en_length = len(en_encoded)
        en_indices = torch.full((self.max_length,), self.pad_id)
        en_indices[:en_length] = torch.tensor(en_encoded)

        return de_indices, en_indices, de_length, en_length
    

