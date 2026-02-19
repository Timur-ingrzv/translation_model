from tqdm.auto import tqdm
from config import TrainConfig, DataConfig
import numpy as np
import torch 

def train_epoch(
        model, 
        optimizer,
        train_loader, 
        loss_fn, 
        device,
        grad_clip=1
    ):
    model.train()
    train_loss = 0.0
    total_tokens = 0
    bar = tqdm(train_loader, 'Train epoch') if TrainConfig.verbose else train_loader
    for de_indices, en_indices, de_length, en_length in bar:
        optimizer.zero_grad()
        de_indices = de_indices[:, :de_length.max()].to(device)
        en_indices = de_indices[:, :en_length.max()].to(device)

        logits = model(de_indices, en_indices[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), en_indices[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        with torch.no_grad():
            non_pad_tokens = (en_indices[:, 1:] == model.pad_id).sum().item()
            train_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    train_loss /= total_tokens
    return train_loss

        
