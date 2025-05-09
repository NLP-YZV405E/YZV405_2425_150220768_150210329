from __init__ import *
from bert_embedder import BERTEmbedder
from pprint import pprint
import torch.nn as nn

class IdiomExtractor(nn.Module):
    def __init__(self,
                 bert_model,
                 bert_config,
                 hparams,
                 device,
                 bert_frozen = True):
        super(IdiomExtractor, self).__init__()
        pprint(hparams)

        self.bert_model = bert_model
        # i changed here because we allredy have a embedder
        self.bert_config = bert_config
        self.hparams = hparams
        self.device = device
        self.bert_frozen = bert_frozen
   
        # Calculate LSTM output dimension first
        self.lstm_output_dim = self.hparams.hidden_dim * 1 if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2

        # batch normalization layers
        self.bn_emb = nn.BatchNorm1d(self.bert_config.hidden_size)
        self.bn_lstm = nn.BatchNorm1d(self.lstm_output_dim)

        # dropout layer
        self.dropout = nn.Dropout(hparams.dropout)

        # we use a bidirectional LSTM
        self.lstm = nn.LSTM(self.bert_config.hidden_size,
                            self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional, 
                            num_layers=self.hparams.num_layers,
                            dropout=self.hparams.dropout if self.hparams.num_layers>1 else 0,
                            batch_first=True)
        
        # linear layer to project the output of the LSTM to the number of classes
        self.classifier = nn.Linear(self.bert_config.hidden_size, hparams.num_classes)

        # we use a CRF layer to model the dependencies between the labels
        self.CRF = CRF(hparams.num_classes,batch_first=True).cuda()
 
    def freeze_bert(self):
        """
        Freezes all parameters in the BERT model to prevent them from being updated during training.
        Call this method to use BERT only as a feature extractor.
        """
        for param in self.bert_model.parameters():
            param.requires_grad = False
        print("BERT model parameters have been frozen.")
    
    def unfreeze_bert(self):
        """
        Unfreezes all parameters in the BERT model to allow them to be updated during training.
        """
        for param in self.bert_model.parameters():
            param.requires_grad = True
        print("BERT model parameters have been unfrozen.")
    
    def forward(self, bert_embeddings, labels):

        if self.bert_frozen:
            self.freeze_bert()
        
        else:
            self.unfreeze_bert()

        # paddlenmiş kısımlara attande etmemek için mask oluşturuyoruz
        mask = self.padding_mask(labels)
        
        # Apply batch norm before dropout (need to reshape for BatchNorm1d)
        batch_size, seq_len, hidden_size = bert_embeddings.shape
        bert_embeddings_reshaped = bert_embeddings.reshape(-1, hidden_size)
        bert_embeddings_norm = self.bn_emb(bert_embeddings_reshaped)
        bert_embeddings = bert_embeddings_norm.reshape(batch_size, seq_len, hidden_size)
        
        bert_embeddings = self.dropout(bert_embeddings)

        X, (h, c) = self.lstm(bert_embeddings)
        
        # Apply batch norm to LSTM output
        batch_size, seq_len, hidden_size = X.shape
        X_reshaped = X.reshape(-1, hidden_size)
        X_norm = self.bn_lstm(X_reshaped)
        X = X_norm.reshape(batch_size, seq_len, hidden_size)
        
        X = self.dropout(X)
 
        O = self.classifier(bert_embeddings)

        if labels==None:
            log_likelihood = -100
        else:
            log_likelihood = self.CRF.forward(O, labels, mask)

        return log_likelihood, O


    def padding_mask(self, labels: torch.Tensor) -> torch.Tensor:

        # create a bool mask: True for real tokens, False for padding
        mask = labels.ne(0)            # shape: (batch, seq_len), dtype=bool
        # CRF requires the first timestep unmasked
        mask[:, 0] = True
        return mask    