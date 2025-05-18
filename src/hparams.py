from __init__ import *
class HParams():
    # Model Architecture Parameters
    dropout = 0.5
    num_classes = 4
    bidirectional = True
    num_layers = 3
    use_lstm = True
    use_attention = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training Parameters
    batch_size = 32
    lr = 0.001
    epoch = 50
    warmup_steps = 1000
    weight_decay = 0.001
    gradient_clip = 1
    scheduler_factor = 0.5
    scheduler_patience = 3
    focal_loss_weight = 0.3 