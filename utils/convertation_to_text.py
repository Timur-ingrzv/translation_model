from typing import List

def convert_to_text(seq_ids: List[int], vocab):
    text = []
    for _id in seq_ids:
        if ((_id == vocab.pad_id) or (_id == vocab.bos_id)):
            continue
        
        if _id == vocab.eos_id:
            break

        text.append(vocab.decode(_id).strip())
    text = ' '.join(text)
    return text