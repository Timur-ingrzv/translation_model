from torch import nn
import torch
import math
from torch.functional import F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TranslationModel(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size, 
            d_model, 
            dropout_rate, 
            num_encoder_layers,
            num_decoder_layers,
            max_length
        ):
        self.pad_id=0
        self.unk_id=1
        self.bos_id=2
        self.eos_id=3

        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=self.pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=self.pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout_rate, max_len=max_length)
        self.transformer = nn.Transformer(
            d_model=d_model,
            dropout=dropout_rate,
            batch_first=True,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            activation='gelu'
        )
        self.max_length = max_length
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.d_model = d_model
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, len_tokens: int, device):
        mask = torch.full((len_tokens, len_tokens), float('-inf')).to(device)
        mask = torch.triu(mask, diagonal=1)
        return mask
    
    def forward(self, src_tokens: torch.Tensor, tgt_tokens_in: torch.Tensor) -> torch.Tensor:
        # src_tokens: (B, src_len)
        # tgt_tokens_in: (B, tgt_len)
        device = src_tokens.device
        src_key_padding_mask = (src_tokens == self.pad_id)
        tgt_key_padding_mask = (tgt_tokens_in == self.pad_id)
        src_embeds = self.src_embedding(src_tokens) * math.sqrt(self.d_model)
        tgt_embeds = self.tgt_embedding(tgt_embeds) * math.sqrt(self.d_model)
        src_embeds = self.pos_enc(src_embeds)
        tgt_embeds = self.pos_enc(tgt_embeds)
        tgt_mask = self.generate_square_subsequent_mask(len(tgt_tokens_in), device)
        outputs = self.transformer(
            src_embeds, tgt_embeds,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        # logits: (B, tgt_len, V)
        logits = self.projection(outputs)
        return logits

    @torch.no_grad()
    def inference(
        self,
        src_tokens: torch.Tensor,
    ) -> torch.Tensor:
        # src_tokens: (B, S)
        self.eval()
        device = src_tokens.device
        B = src_tokens.shape[0]
        tokens = torch.full((B, 1), self.bos_id).to(device)
        is_finished = torch.zeros(B, dtype=torch.bool)
        for step in range(self.max_length):
            # logits (B, tokens_len, V)
            logits = self.forward(src_tokens, tokens)
            #next_token: (B,)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            tokens = torch.concatenate([tokens, next_token], dim=-1)
            is_finished = (is_finished | (next_token == self.eos_id))
            if is_finished.all().item():
                break

        return tokens