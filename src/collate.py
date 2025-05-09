import torch
from __init__ import * 

def collate(elems: tuple) -> tuple:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # unzip the batch of triples
    words, label_seqs, lang_seqs = zip(*elems)

    pad_labels = pad_sequence(label_seqs, batch_first=True, padding_value=0)
    pad_langs  = pad_sequence(lang_seqs,  batch_first=True, padding_value=-1)

    # move both to the device
    pad_labels = pad_labels.to(device)

    # cümlelerin pad öncesi uzunluğu
    seq_lens = [len(w) for w in words]  # örn. [8, 12, 5, …]

    # get sentence level language labels
    sent_langs = []
    for i, L in enumerate(seq_lens):
        if L > 0:
            # get first language label as sentence level label
            lang0 = pad_langs[i, 0].item()
            sent_langs.append(lang0)
        else:
            # boş cümle olursa -1 atayalım
            sent_langs.append(-1)

    sent_langs = torch.tensor(sent_langs, device=device)  # shape == (batch_size,)

    return list(words), pad_labels, sent_langs
