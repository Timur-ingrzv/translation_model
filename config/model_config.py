from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ModelConfig:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 32
    embed_dim = 32
    dropout_rate = 0.1