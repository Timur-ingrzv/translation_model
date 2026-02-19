from collections import Counter
from typing import List, Union
import torch
import os
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor

class Vocab:
    def __init__(self, prefix):
        from config import DataConfig
        if not os.path.isfile(f'tokenizer_{prefix}.model'):
            SentencePieceTrainer.train(
                input=f'data/train.de-en.{prefix}',
                model_prefix=f'tokenizer_{prefix}',
                vocab_size=DataConfig.vocab_size, 
                model_type='word',  
                character_coverage=1.0,
    
                split_by_whitespace=True,  # разбивать по пробелам
                split_digits=False,  # не разбивать цифры
                split_by_number=False,  # не разбивать по числам
                split_by_unicode_script=False, 
    
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece='<PAD>',
                unk_piece='<UNK>',
                bos_piece='<BOS>',
                eos_piece='<EOS>',
    
                add_dummy_prefix=False, 
                remove_extra_whitespaces=False,
                treat_whitespace_as_suffix=False
            )
        self.pad_id=0
        self.unk_id=1
        self.bos_id=2
        self.eos_id=3
        self.vocab = SentencePieceProcessor(model_file=f'tokenizer_{prefix}' + '.model')

    def encode(self, texts: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        return self.vocab.encode(texts)

    def decode(self, ids: Union[torch.Tensor, List[int], List[List[int]]]) -> Union[str, List[str]]:
        if torch.is_tensor(ids):
            assert len(ids.shape) <= 2, 'Expected tensor of shape (length, ) or (batch_size, length)'
            ids = ids.cpu().tolist()

        return self.vocab.decode(ids)
