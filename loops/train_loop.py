from tqdm.auto import tqdm
from config import TrainConfig, DataConfig
import numpy as np
import torch 

def train_epoch(
        encoder,
        decoder,
        optimizer_encoder,
        optimizer_decoder,
        train_loader, 
        loss_fn, 
        device
    ):
    encoder.train()
    decoder.train()
    train_loss = 0.0
    bar = tqdm(train_loader, 'Train epoch') if TrainConfig.verbose else train_loader
    for de_indices, en_indices, de_length, en_length in bar:
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        de_indices = de_indices[:, :de_length.max()].to(device)
        encoder_outputs, encoder_hidden = encoder(de_indices, de_length)

        en_indices = en_indices[:, :en_length.max()].to(device)
        logits = decoder(en_indices[:, :-1], en_length - 1, encoder_hidden)
        loss = loss_fn(logits.transpose(1, 2), en_indices[:, 1:])

        train_loss += loss.item() * en_indices.shape[0]
        
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

    train_loss /= len(train_loader.dataset)
    return train_loss

        
