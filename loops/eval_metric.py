import torch 
import pandas as pd
from tqdm.auto import tqdm
from config import TrainConfig
import sacrebleu

@torch.no_grad()
def evaluate_bleu(encoder, decoder, val_loader, device):
    predictions = []
    encoder.eval()
    decoder.eval()
    total_bleu = 0.0

    for de_indices, en_indices, de_length, en_length in tqdm(val_loader, desc='Evaluating BLEU'):
        de_indices = de_indices[:, :de_length.max()].to(device)

        encoder_outputs, encoder_hidden = encoder(de_indices, de_length)
        tokens_ids = decoder.inference(encoder_hidden).cpu().tolist()
        predictions.extend([el.strip() for el in decoder.en_vocab.decode(tokens_ids)])
    
        labels = decoder.en_vocab.decode(en_indices)
        labels = [[label.strip()] for label in labels]

        bleu = sacrebleu.corpus_bleu(predictions, labels)
        total_bleu += bleu.score * de_indices.shape[0]

    return total_bleu / len(val_loader.dataset)