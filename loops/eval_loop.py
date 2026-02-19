import torch
from tqdm.auto import tqdm
from config import TrainConfig

@torch.no_grad()
def eval_epoch(model, loss_fn, val_loader, device):
    val_loss = 0.0
    total_tokens = 0
    model.eval()
    bar = tqdm(val_loader, 'Eval epoch') if TrainConfig.verbose else val_loader
    for de_indices, en_indices, de_length, en_length in bar:
        de_indices = de_indices[:, :de_length.max()].to(device)
        en_indices = en_indices[:, :en_length.max()].to(device)

        logits = model(de_indices, en_indices[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), en_indices[:, 1:].reshape(-1))
        non_pad_tokens = (en_indices[:, 1:] != model.pad_id).sum()
        val_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

    val_loss /= total_tokens
    return val_loss
