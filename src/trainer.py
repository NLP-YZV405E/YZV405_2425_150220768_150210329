import model
import math
from __init__ import *

class Trainer():
    def __init__(self,
                model:nn.Module, 
                optimizer,
                labels_vocab):
        
        self.model = model
        self.optimizer = optimizer
        self.labels_vocab = labels_vocab

    def padding_mask(self, batch):
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        padding = padding.type(torch.uint8)
        return padding.to(torch.bool)
 
    def train(self,
            train_dataset:Dataset, 
            valid_dataset:Dataset,
            epochs:int=1,
            patience:int=10,
            modelname="idiom_expr_detector"):
        
        print("\nTraining...")
 
        
        train_loss_list = []
        dev_loss_list = []
        f1_scores = []
        # best f1 score
        record_dev = 0.0
        
        full_patience = patience
        
        modelname = modelname

        for epoch in range(epochs):
            if patience <= 0:
                print("Stopping early (no more patience).")
                break

            print(" Epoch {:03d}".format(epoch + 1))

            train_loss = 0.0
            self.model.train()
            
            count_batches = 0
            self.optimizer.zero_grad()
            
            for words, labels, lang in tqdm(train_dataset):
                count_batches+=1
                
                # add here language check

                batch_LL, _ = self.model(words, labels)
                # get negative log likelihood
                batch_NLL = - torch.sum(batch_LL)/8

                loss_val = batch_NLL.item()

                # bazen batch_NLL NaN olabiliyor, bu durumda loss'u hesaplamıyoruz
                if not math.isnan(loss_val):
                    # calculate backpropagation
                    batch_NLL.backward()
                    # clip gradients to avoid exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    # save the loss value for graph
                    train_loss += loss_val
                    # update weights
                    self.optimizer.step()
                    # clear gradients
                    self.optimizer.zero_grad()


            avg_train_loss = train_loss / len(train_dataset)
            train_loss_list.append(avg_train_loss)
            print('[E: {:2d}] train loss = {:0.4f}'.format(epoch+1, avg_train_loss))

            valid_loss, f1 = self.evaluate(valid_dataset)
            f1_scores.append(f1)


            # save the model if the f1 score is better than the previous best
            if f1>record_dev:
                record_dev = f1
                torch.save(self.model.state_dict(), r"./checkpoints/"+modelname+".pt")
                patience = full_patience
            else:
                patience -= 1
            
            print('\t[E: {:2d}] valid loss = {:0.4f}, f1-score = {:0.4f}, patience: {:2d}'.format(epoch+1, valid_loss, f1, patience))
            dev_loss_list.append(valid_loss)

        print("...Done!")
        return train_loss_list, dev_loss_list, f1_scores 

    def evaluate(self, valid_dataset):

        valid_loss = 0.0
        all_predictions = list()
        all_labels = list()
        labels_vocab_reverse = {v:k for (k,v) in self.labels_vocab.items()}
         
        self.model.eval()
    
        for words, labels, lang in tqdm(valid_dataset):

            self.optimizer.zero_grad()

            with torch.no_grad():
                batch_LL, predictions = self.model(words, labels)

            batch_NLL = - torch.sum(batch_LL)/8

            val_loss = batch_NLL.item()

            predictions = predictions.view(-1, predictions.shape[-1])
            labels = labels.view(-1) 
 
            for i in range(len(predictions)):
                if labels[i]!=0:
                    current_prdiction = int(torch.argmax(predictions[i]))
                    all_predictions.append(labels_vocab_reverse[current_prdiction])
                    all_labels.append(labels_vocab_reverse[int(labels[i])])
            
            if not math.isnan(val_loss):
                valid_loss += val_loss

        f1 = f1_score(all_labels, all_predictions, average= 'macro')
        print(classification_report(all_labels, all_predictions, digits=3))
        print(f1)
        
        return valid_loss / len(valid_dataset), f1