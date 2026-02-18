import torch
from tqdm.auto import tqdm
from config import TrainConfig

@torch.no_grad()
def eval_epoch(encoder, decoder, loss_fn, valid_loader, device):
    valid_loss = 0.0
    encoder.eval()
    decoder.eval()
    bar = tqdm(valid_loader, 'Eval epoch') if TrainConfig.verbose else valid_loader
    for de_indices, en_indices, de_length, en_length in bar:
        de_indices = de_indices[:, :de_length.max()].to(device)
        en_indices = en_indices[:, :en_length.max()].to(device)

        encoder_outputs, encoder_hidden = encoder(de_indices, de_length)
        logits = decoder(en_indices[:, :-1], en_length - 1, encoder_hidden)
        loss = loss_fn(logits.transpose(1, 2), en_indices[:, 1:])

        valid_loss += loss.item() * de_indices.shape[0]
        
    valid_loss /= len(valid_loader.dataset)
    return valid_loss
