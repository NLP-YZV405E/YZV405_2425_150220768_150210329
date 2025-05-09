from __init__ import *
from bert_embedder import BERTEmbedder
from pprint import pprint
import torch.nn as nn
import torch.nn.functional as F

class IdiomExtractor(nn.Module):
    def __init__(self,
                 bert_model,
                 bert_config,
                 hparams,
                 device,
                 useLSTM=False):
        super(IdiomExtractor, self).__init__()
        pprint(hparams)

        self.bert_model = bert_model
        self.bert_config = bert_config
        self.hparams = hparams
        self.device = device
        self.useLSTM = useLSTM

        # lstm and bert output dimension will be the same

        # batch normalization layers 
        self.bn_emb = nn.BatchNorm1d(self.bert_config.hidden_size) # hidden size of 768
        self.bn_lstm = nn.BatchNorm1d(self.bert_config.hidden_size)

        # dropout layer with 0.5 dropout rate
        self.dropout = nn.Dropout(hparams.dropout)

        # use bidirectional lstm
        self.lstm = nn.LSTM(self.bert_config.hidden_size,
                            self.bert_config.hidden_size // 2 if hparams.bidirectional else self.bert_config.hidden_size,
                            bidirectional=self.hparams.bidirectional, 
                            num_layers=self.hparams.num_layers,
                            dropout=self.hparams.dropout if self.hparams.num_layers>1 else 0,
                            batch_first=True)
        

        self.classifier = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(hparams.dropout),
            nn.Linear(128, hparams.num_classes),
        )

        # we use a CRF layer to model the dependencies between the labels
        self.CRF = CRF(hparams.num_classes, batch_first=True).cuda()
 
    def freeze_bert(self):
        for param in self.bert_model.parameters():
            param.requires_grad = False
        print("BERT model parameters have been frozen.")
    
    def unfreeze_bert(self):
        for param in self.bert_model.parameters():
            param.requires_grad = True
        print("BERT model parameters have been unfrozen.")
    
    def forward(self, bert_embeddings, labels):

        # paddlenmiş kısımlara attande etmemek için mask oluşturuyoruz
        mask = self.padding_mask(labels)
        
        # Apply batch norm before dropout (need to reshape for BatchNorm1d)
        batch_size, seq_len, hidden_size = bert_embeddings.shape
        X = nn.functional.layer_norm(bert_embeddings, normalized_shape=[hidden_size])
        
        # bert embeddingse dropout uygulamak saçma geldi
        #bert_embeddings = self.dropout(bert_embeddings)

        # Apply LSTM to the BERT embeddings
        if self.useLSTM:
            X, (h, c) = self.lstm(X)
            # Apply batch norm to LSTM output
            batch_size, seq_len, hidden_size = X.shape
            X = nn.functional.layer_norm(X, normalized_shape=[hidden_size])
            # Apply dropout to LSTM output
            X = self.dropout(X)

        # Apply linear layer
        O = self.classifier(X)

        if labels is None:
            log_likelihood = -100
        else:
            log_likelihood = self.CRF.forward(O, labels, mask)

        return log_likelihood, O


    def padding_mask(self, labels: torch.Tensor) -> torch.Tensor:

        # create a bool mask: True for real tokens, False for padding
        mask = labels.ne(-1)            # shape: (batch, seq_len), dtype=bool
        # CRF requires the first timestep unmasked
        mask[:, 0] = True
        return mask    