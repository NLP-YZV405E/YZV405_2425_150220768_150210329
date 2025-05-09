from __init__ import *
class HParams():
    dropout = 0.5
    num_classes = 4
    bidirectional = True
    num_layers = 2
    use_lstm = True
    device = "cuda" if torch.cuda.is_available() else "cpu"