from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class ModelConfig:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    d_model = 128
    dropout_rate = 0.1
    num_decoder_layers = 4
    num_encoder_layers = 6
    dim_feedforward = 1024