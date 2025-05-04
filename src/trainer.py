import model
import math
from __init__ import *

class Trainer():
    def __init__(self,
                tr_model:nn.Module,
                it_model:nn.Module, 
                tr_optimizer,
                it_optimizer,
                labels_vocab):
        
        self.tr_model = tr_model,
        self.it_model = it_model,
        self.tr_optimizer = tr_optimizer,
        self.it_optimizer = it_optimizer,
        self.labels_vocab = labels_vocab

    def padding_mask(self, batch):
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        padding = padding.type(torch.uint8)
        return padding.to(torch.bool)
 
    def train(self,
            train_dataset:DataLoader, 
            valid_dataset:DataLoader,
            epochs:int=1,
            patience:int=10,
            modelname: str = "idiom_expr_detector"):
        
        print("\nTraining...")
 
        
        train_loss_list = []
        dev_loss_list = []
        f1_scores = []
        # best f1 score
        record_dev = 0.0
        
        full_patience = patience
        

        for epoch in range(epochs):
            if patience <= 0:
                print("Stopping early (no more patience).")
                break

            print(" Epoch {:03d}".format(epoch + 1))

            self.tr_model.train()
            self.it_model.train()

            tr_loss = it_loss = total_loss = 0.0

            tr_batches = it_batches = 0
                
            for words, labels, langs in tqdm(train_dataset):

                tr_mask = (langs == 0).to(torch.bool)
                it_mask = (langs == 1).to(torch.bool)

                loss = 0
                parts = 0

                if tr_mask.any():
                    # get the batch for the tr_model
                    words_tr = words[tr_mask]
                    labels_tr = labels[tr_mask]

                    # get the loss and predictions
                    tr_LL, _ = self.tr_model(words_tr, labels_tr)
                    tr_NLL = - torch.sum(tr_LL)/ words_tr.size(0)

                    tr_loss += tr_NLL.item()
                    loss += tr_NLL
                    parts += 1
                    tr_batches += 1


                if it_mask.any():
                    # get the batch for the it_model
                    words_it = words[it_mask]
                    labels_it = labels[it_mask]
                    # get the loss and predictions
                    it_LL, _ = self.it_model(words_it, labels_it)
                    it_NLL = - torch.sum(it_LL)/ words_it.size(0)
                    it_loss += it_NLL.item()
                    loss += it_NLL
                    parts += 1
                    it_batches += 1

                loss = loss / parts
                self.tr_optimizer.zero_grad()
                self.it_optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.tr_model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.it_model.parameters(), 1)

                self.tr_optimizer.step()
                self.it_optimizer.step()

                total_loss += loss.item()

                

            avg_tr = tr_loss / tr_batches if tr_batches > 0 else 0
            avg_it = it_loss / it_batches if it_batches > 0 else 0
            train_loss = total_loss / (tr_batches+it_batches)

            train_loss_list.append(train_loss)
            print('[E: {:2d}] train loss = {:0.4f}, tr_loss = {:0.4f}, it_loss = {:0.4f}'.format(epoch+1, train_loss, avg_tr, avg_it))

            valid_loss, f1 = self.evaluate(valid_dataset)
            dev_loss_list.append(valid_loss)
            f1_scores.append(f1)


            # save the model if the f1 score is better than the previous best
            if f1>record_dev:
                record_dev = f1
                torch.save({
                    "tr_model": self.tr_model.state_dict(),
                    "it_model": self.it_model.state_dict(),
                }, r"./src/checkpoints/"+modelname+".pt")
                patience = full_patience
            else:
                patience -= 1
            
            print(f"\t[E:{epoch:02d}] valid_loss={valid_loss:.4f}  f1={f1:.4f}  patience={patience}")

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