from typing import List

def convert_to_text(seq_ids: List[int], vocab):
    filtered_text = []
    for _id in seq_ids:
        if ((_id == vocab.pad_id) or (_id == vocab.bos_id)):
            continue

        if _id == vocab.eos_id:
            break
        
        filtered_text.append(vocab.decode(_id).strip())

    return ' '.join(filtered_text).strip()