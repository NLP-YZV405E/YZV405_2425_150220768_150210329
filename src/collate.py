import torch
from __init__ import * 

def collate(elems: tuple) -> tuple:
    # Determine if CUDA is available (for GPU) or fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    words, labels = list(zip(*elems))
    pad_labels = pad_sequence(labels, batch_first=True, padding_value=0)
 
    # Berke changed this code, it was: return list(words), pad_labels.cuda()
    return list(words), pad_labels.to(device)