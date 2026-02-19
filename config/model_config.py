from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ModelConfig:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    d_model = 256
    dropout_rate = 0.1
    num_decoder_layers = 6
    num_encoder_layers = 8