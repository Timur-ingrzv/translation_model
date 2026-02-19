import torch 
import pandas as pd
from tqdm.auto import tqdm
from config import TrainConfig
import sacrebleu
from utils import convert_to_text

@torch.no_grad()
def evaluate_bleu(model, val_loader, device):
    predictions = []
    model.eval()
    total_bleu = 0.0
    en_vocab = val_loader.dataset.en_vocab

    for de_indices, en_indices, de_length, en_length in tqdm(val_loader, desc='Evaluating BLEU'):
        de_indices = de_indices[:, :de_length.max()].to(device)
        en_indices = en_indices[:, :en_length.max()].to(device)

        predicted_tokens = model.inference(de_indices)
        predictions = [convert_to_text(seq, en_vocab) for seq in predicted_tokens]
        labels = en_vocab.decode(en_indices)
        labels = [[label.strip()] for label in labels]

        bleu = sacrebleu.corpus_bleu(predictions, labels)
        total_bleu += bleu.score * de_indices.shape[0]

    return total_bleu / len(val_loader.dataset)