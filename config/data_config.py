from dataclasses import dataclass
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import TextTrainDataset, TextTestDataset
import os 
import sys

@dataclass
class DataConfig:
    root = 'data'
    val_size = 0.2
    min_frequency = 2
    max_length = 60
    vocab_size = 50_000

    with open(os.path.join(root, 'train.de-en.de')) as f:
        de_texts = f.read().splitlines()
    with open(os.path.join(root, 'train.de-en.en')) as f:
        en_texts = f.read().splitlines()

    texts_train = list(zip(de_texts, en_texts))
    texts_train, _ = train_test_split(texts_train, test_size=0.4)
    
    with open(os.path.join(root, 'val.de-en.de')) as f:
        de_texts = f.read().splitlines()
    with open(os.path.join(root, 'val.de-en.en')) as f:
        en_texts = f.read().splitlines()

    texts_val = list(zip(de_texts, en_texts))

    del de_texts
    del en_texts
    def get_train_dataset(self):
        return TextTrainDataset(self.texts_train)
    
    def get_val_dataset(self, de_vocab, en_vocab):
        return TextTrainDataset(self.texts_val, de_vocab, en_vocab)
    
    def get_test_dataset(self,  de_vocab, en_vocab):
        return TextTestDataset(os.path.join(self.root, 'test1.de-en.de'),  de_vocab, en_vocab)
