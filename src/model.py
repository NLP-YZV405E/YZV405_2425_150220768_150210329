from __init__ import *
from bert_embedder import BERTEmbedder
from pprint import pprint
import torch.nn as nn
import torch.nn.functional as F

class IdiomExtractor(nn.Module):
    def __init__(self,
                 bert_model,
                 hparams):
        super(IdiomExtractor, self).__init__()
        pprint(hparams)

        self.bert = bert_model
        self.use_lstm = hparams.use_lstm
        self.hidden_size = bert_model.config.hidden_size

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
 
    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        print("BERT model parameters have been frozen.")
    
    def unfreeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = True
        print("BERT model parameters have been unfrozen.")
    
    def forward(self, bert_embeddings, labels):

        # trainde label olacak ve pad edilmiş kısımlara attend etmemek için mask oluşturuyoruz
        if labels is not None:
            # padding olan kısımlara attande etmemek için mask oluşturuyoruz
            mask = labels.ne(0)
        # test kısmında label olmayak bu yüzden tüm kısımlara attend etmemiz lazım
        else:
            mask = torch.ones(bert_embeddings.shape[:2], dtype=torch.bool,
                              device=bert_embeddings.device)
        # CRF ilk time‐step’in unmasked olmasını ister
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
        emissions  = self.classifier(X)
        
        # print(f"emissions shape: {emissions.shape}")
        # emissions shape: torch.Size([7, 14, 4])

        # eğer label yoksa (inference) decode etmemiz lazım
        if labels is None:
            # return highest probability sequence
            return self.CRF.decode(emissions, mask=mask)
        
        
        # log likelihood loss fonksiyonumuz, burdan backward ile backpropagate ediyoruz
        log_likelihood = self.CRF(emissions, labels, mask)
        print(f"log_likelihood shape: {log_likelihood.shape}")
        print(f"log_likelihood: {log_likelihood}")

        # print(f"log_likelihood shape: {log_likelihood.shape}")
        # tensor(-102.7468, device='cuda:0', grad_fn=<SumBackward0>)

        return log_likelihood, emissions
    
    