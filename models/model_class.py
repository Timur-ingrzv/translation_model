from torch import nn
import torch
from torch.functional import F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor):
        embeds = self.embedding(indices)
        embeds = self.dropout(embeds)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed_embeds)
        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden
    

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, en_vocab, max_length):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.max_length = max_length
        self.en_vocab = en_vocab

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor, encoder_hidden: torch.Tensor) -> torch.Tensor:
        # embeds (B, L, d)
        embeds = self.embedding(indices)
        packed_embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        # outputs (B, L, hidden)
        # hidden (num_layers * bidir, B, hidden)
        decoder_hidden = encoder_hidden
        outputs, decoder_hidden = self.gru(packed_embeds, decoder_hidden)
        outputs, lengths = pad_packed_sequence(outputs, batch_first=True)
        # logits (B, L, Vocab_size)
        logits = self.out(outputs)
        return logits
    
    @torch.inference_mode()
    def inference(self, encoder_output) -> str:
        self.eval()

        device = next(self.parameters()).device
        # (B, 1)
        B = encoder_output.shape[1]
        tokens = torch.full((B, 1), 2, device=device)
        embeds = self.embedding(tokens)

        #outputs (B, 1, hidden)
        outputs, h = self.gru(embeds, encoder_output)
        
        logits = self.out(outputs)
        # new_token (B, 1)
        new_token = logits.argmax(dim=-1)
        tokens = torch.cat([tokens, new_token], dim=1)

        while tokens.shape[1] < self.max_length:            
            embeds = self.embedding(new_token)

            outputs, h = self.gru(embeds, h)
            logits = self.out(outputs)
            new_token = logits.argmax(dim=-1)
            tokens = torch.cat([tokens, new_token], dim=1)

        return tokens
