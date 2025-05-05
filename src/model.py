from __init__ import *
from bert_embedder import BERTEmbedder

class IdiomExtractor(nn.Module):
    def __init__(self,
                 bert_model,
                 bert_tokenizer,
                 bert_config,
                 hparams,
                 device):
        super(IdiomExtractor, self).__init__()
        pprint(hparams)

        self.bert_model = bert_model
        # i changed here because we allredy have a embedder
        self.embedder = BERTEmbedder(bert_model, bert_tokenizer, device)
        self.bert_config = bert_config
        self.hparams = hparams
        self.device = device
   
        # dropout layer
        self.dropout = nn.Dropout(hparams.dropout)

        # we use a bidirectional LSTM
        self.lstm = nn.LSTM(self.bert_config.hidden_size,
                            self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional, 
                            num_layers=self.hparams.num_layers,
                            dropout=self.hparams.dropout if self.hparams.num_layers>1 else 0,
                            batch_first=True)
        
        # we set the output dimension of the LSTM to be the same as the input dimension
        self.lstm_output_dim = self.hparams.hidden_dim * 1 if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2

        # linear layer to project the output of the LSTM to the number of classes
        self.classifier = nn.Linear(self.bert_config.hidden_size, hparams.num_classes)

        # we use a CRF layer to model the dependencies between the labels
        self.CRF = CRF(hparams.num_classes,batch_first=True).cuda()
 
 
    def forward(self, bert_embeddings, labels):

        # paddlenmiş kısımlara attande etmemek için mask oluşturuyoruz
        mask = self.padding_mask(labels)
        
        bert_embeddings = self.dropout(bert_embeddings)

        X, (h, c) = self.lstm(bert_embeddings)
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