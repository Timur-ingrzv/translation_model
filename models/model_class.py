from torch import nn
import torch
import math
from torch.functional import F
import torch.utils.data as tud

# https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=200):
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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class TranslationModel(nn.Module):
    def __init__(
            self,
            src_vocab_size=1000,
            tgt_vocab_size=1000, 
            d_model=128, 
            dropout_rate=0.1, 
            num_encoder_layers=2,
            num_decoder_layers=2,
            max_length=100,
            dim_feedforward=256
        ):
        super().__init__()
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
            activation='gelu',
            dim_feedforward=dim_feedforward
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
        tgt_embeds = self.tgt_embedding(tgt_tokens_in) * math.sqrt(self.d_model)
        src_embeds = self.pos_enc(src_embeds)
        tgt_embeds = self.pos_enc(tgt_embeds)
        tgt_mask = self.generate_square_subsequent_mask(tgt_tokens_in.shape[1], device)
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
        beam_width=5
    ) -> torch.Tensor:
        # src_tokens: (B, S)
        self.eval()
        device = src_tokens.device
        B = src_tokens.shape[0]
        output_tokens = torch.full((B, 1), fill_value=self.bos_id).to(device)
        logits = self.forward(src_tokens, output_tokens)
        next_probs = logits[:, -1, :]
        vocab_size = next_probs.shape[-1]
        # probs: (B, beam_width)
        probs, next_tokens = next_probs.squeeze().log_softmax(-1)\
        .topk(k = beam_width, axis = -1)
        output_tokens = output_tokens.repeat((beam_width, 1))
        next_tokens = next_tokens.reshape(-1, 1)
        output_tokens = torch.cat([output_tokens, next_tokens], dim=-1)

        for _ in range(self.max_length-1):
            dataset = tud.TensorDataset(src_tokens.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim = 1), output_tokens)
            loader = tud.DataLoader(dataset, batch_size = 64)
            next_probs = []
            for src, tgt in loader:
                next_probs.append(self.forward(src, tgt)[:, -1, :].log_softmax(-1))
            # next_probs: (B * beam_width, V)
            next_probs = torch.cat(next_probs, dim=0)
            # next_probs: (B, beam_width, V)
            next_probs = next_probs.reshape((-1, beam_width, next_probs.shape[-1]))
            probs = probs.unsqueeze(-1) + next_probs
            probs = probs.flatten(start_dim=1)
            # probs имеет формат beam1 -> [vocab_size], beam2 -> [vocab_size] ...
            probs, next_t = probs.topk(k=beam_width, axis=-1)
            next_tokens = torch.remainder(next_t, vocab_size).flatten().unsqueeze(-1)
            best_candidates = (next_t / vocab_size).long()
            best_candidates += torch.arange(output_tokens.shape[0] // beam_width, device = device).unsqueeze(-1) * beam_width
            output_tokens = output_tokens[best_candidates].flatten(end_dim = -2)
            output_tokens= torch.cat((output_tokens, next_tokens), dim = -1)
           
    
        # best_beam_indices (B,)
        best_beam_idx= probs.argmax(dim=-1) 
        offsets = torch.arange(output_tokens.shape[0] // beam_width, device=device) * beam_width
        global_idx = offsets + best_beam_idx
        best_sequences = output_tokens[global_idx]
        return best_sequences   
        
      

        