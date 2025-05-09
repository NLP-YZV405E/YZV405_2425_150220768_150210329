import torch
from __init__ import * 

def collate(elems: tuple) -> tuple:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # unzip the batch of triples
    words, label_seqs, lang_seqs = zip(*elems)

    pad_labels = pad_sequence(label_seqs, batch_first=True, padding_value=-1)
    pad_langs  = pad_sequence(lang_seqs,  batch_first=True, padding_value=-1)

    # move both to the device
    pad_labels = pad_labels.to(device)
    pad_langs  = pad_langs.to(device)

    return list(words), pad_labels, pad_langs
