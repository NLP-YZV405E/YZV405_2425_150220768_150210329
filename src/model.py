from __init__ import *
from bert_embedder import BERTEmbedder
from pprint import pprint
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class IdiomExtractor(nn.Module):
    def __init__(self,
                 bert_embedder,
                 hparams):
        super(IdiomExtractor, self).__init__()
        pprint(hparams)

        self.hidden_size = bert_embedder.bert_model.config.hidden_size
        self.bert_embedder = bert_embedder
        self.use_lstm = hparams.use_lstm
        self.device = hparams.device
        self.focal_loss_weight = hparams.focal_loss_weight

        # lstm and bert output dimension will be the same
        if self.use_lstm:
            self.lstm = nn.LSTM(
                self.hidden_size, # input size = hidden size of bert -> 768
                # output size = hidden size of lstm -> 768
                self.hidden_size // 2 if hparams.bidirectional else self.hidden_size,
                batch_first=True, # shape = (batch, seq_len, hidden_size)
                bidirectional=hparams.bidirectional,
                num_layers=hparams.num_layers,
                dropout=hparams.dropout if hparams.num_layers > 1 else 0,
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(128, hparams.num_classes),
        )

        # dropout layer with 0.5 dropout rate
        self.dropout = nn.Dropout(hparams.dropout)

        # we use a CRF layer to model the dependencies between the labels
        self.CRF = CRF(hparams.num_classes, batch_first=True).cuda()
        
        # Initialize focal loss
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
 
    def freeze_bert(self):
        for param in self.bert_embedder.bert_model.parameters():
            param.requires_grad = False
        print("BERT model parameters have been frozen.")
    
    def unfreeze_bert(self):
        for param in self.bert_embedder.bert_model.parameters():
            param.requires_grad = True
        print("BERT model parameters have been unfrozen.")
    
    def forward(self, sents, labels, seq_len=None):
        # cümleleri embed et
        bert_embeddings = self.bert_embedder.embed_sentences(sents)
        # cümleleri paddingle
        bert_embeddings = pad_sequence(bert_embeddings, batch_first=True, padding_value=0).to(self.device)
        # eğer tr dilindeki cümlelerin uzunluğu seq_len'den küçükse, seq_len'e pad et
        if bert_embeddings.size(1) < seq_len:
            pad_amt = seq_len - bert_embeddings.size(1)
            bert_embeddings = F.pad(bert_embeddings, (0, 0, 0, pad_amt), "constant", 0)

        # trainde label olacak ve pad edilmiş kısımlara attend etmemek için mask oluşturuyoruz
        if labels is not None:
            # padding olan kısımlara attande etmemek için mask oluşturuyoruz
            mask = labels.ne(0)
        # test kısmında label olmayak bu yüzden tüm kısımlara attend etmemiz lazım
        else:
            mask = torch.ones(bert_embeddings.shape[:2], dtype=torch.bool,
                              device=bert_embeddings.device)
        # CRF ilk time‐step'in unmasked olmasını ister
        mask[:, 0] = True
        
        # Apply batch norm before dropout (need to reshape for BatchNorm1d)
        
        # apply dropout -> bu saçma değilmiş gerekliymiş
        # ayrıca bertin kendi içinde layernormu var o yüzen layer norma gerek yok
        X = self.dropout(bert_embeddings)

        if self.use_lstm:
            X,_ = self.lstm(X)
            X = nn.functional.layer_norm(X, normalized_shape=[self.hidden_size])
            X = self.dropout(X)
        
        # Apply linear layer, 
        emissions = self.classifier(X)
        
        # eğer label yoksa (inference) decode etmemiz lazım
        if labels is None:
            # return highest probability sequence
            return self.CRF.decode(emissions, mask=mask)
        
        # CRF loss
        crf_loss = -self.CRF(emissions, labels, mask)
        
        # Focal loss
        # Reshape for focal loss calculation
        emissions_flat = emissions.view(-1, emissions.size(-1))
        labels_flat = labels.view(-1)
        mask_flat = mask.view(-1)
        
        # Calculate focal loss only on non-padded tokens
        focal_loss = self.focal_loss(
            emissions_flat[mask_flat],
            labels_flat[mask_flat]
        )
        
        # Combine losses
        total_loss = crf_loss + self.focal_loss_weight * focal_loss
        
        return total_loss, emissions
    
    