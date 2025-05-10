import model
import math
from __init__ import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

class Trainer:
    def __init__(self,
                 tr_model: nn.Module,
                 it_model: nn.Module,
                 tr_optimizer,
                 it_optimizer,
                 tr_embedder,
                 it_embedder,
                 labels_vocab: dict,
                 modelname = "idiom_expr_detector"):
        self.tr_model     = tr_model
        self.it_model     = it_model
        self.tr_optimizer = tr_optimizer
        self.it_optimizer = it_optimizer
        self.labels_vocab = labels_vocab
        self.tr_embedder  = tr_embedder
        self.it_embedder  = it_embedder
        self.modelname    = modelname
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"

    def padding_mask(self, batch: torch.Tensor) -> torch.BoolTensor:
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        return padding.type(torch.bool)

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              epochs: int = 20,
              patience: int = 10):

        print("\nTraining...")
        train_loss_list = []
        dev_acc_list    = []
        f1_scores       = []
        record_dev      = 0.0
        full_patience   = patience

        for epoch in range(1, epochs + 1):
            if patience <= 0:
                print("Stopping early (no more patience).")
                break

            print(f" Epoch {epoch:03d}, patience: {patience}")
            self.tr_model.train()
            self.it_model.train()

            # epochtaki loss ve batch sayıları
            tr_loss_sum = it_loss_sum = 0.0
            tr_batches  = it_batches  = 0

            for words, labels, langs in tqdm(train_loader, desc=f"Epoch {epoch}"):

                batch_size, seq_len = labels.shape
                device = labels.device
                hidden_size = self.tr_embedder.bert_model.config.hidden_size

                # get indices of turkish and italian sentences
                tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
                it_indices = (langs == 1).nonzero(as_tuple=True)[0]

                # eğer tr dilinde cümle varsa
                if len(tr_indices) > 0:
                    # tr dilindeki cümlelerin labels'ını al
                    tr_labels = labels[tr_indices]
                    # tr dilindeki cümleleri al
                    tr_sents = [words[i] for i in tr_indices.cpu().numpy()]
                    # cümleleri embed et
                    tr_embeds = self.tr_embedder.embed_sentences(tr_sents)
                    # cümleleri paddingle
                    tr_embs = pad_sequence(tr_embeds, batch_first=True, padding_value=0).to(device)
                    # eğer tr dilindeki cümlelerin uzunluğu seq_len'den küçükse, seq_len'e pad et
                    if tr_embs.size(1) < seq_len:
                        pad_amt = seq_len - tr_embs.size(1)
                        tr_embs = F.pad(tr_embs, (0, 0, 0, pad_amt), "constant", 0)
                else:
                    # eğer tr dilinde cümle yoksa, 0'larla doldur
                    tr_embs = torch.zeros((0, seq_len, hidden_size), device=device)

                if len(it_indices) > 0:
                    it_labels = labels[it_indices]
                    it_sents  = [words[i] for i in it_indices.cpu().numpy()]
                    it_embeds = self.it_embedder.embed_sentences(it_sents)
                    it_embs   = pad_sequence(it_embeds, batch_first=True, padding_value=0).to(device)
                    if it_embs.size(1) < seq_len:
                        pad_amt = seq_len - it_embs.size(1)
                        it_embs = F.pad(it_embs, (0, 0, 0, pad_amt), "constant", 0)
                else:
                    it_embs = torch.zeros((0, seq_len, hidden_size), device=device)

                tr_NLL = 0
                it_NLL = 0

                if len(tr_indices) > 0:
                    tr_LL, _ = self.tr_model(tr_embs, tr_labels)
                    tr_NLL = -tr_LL
                    tr_batches += 1
                
                if len(it_indices) > 0:
                    it_LL, _ = self.it_model(it_embs, it_labels)
                    it_NLL = -it_LL
                    it_batches += 1


                loss = tr_NLL + it_NLL

                tr_loss_sum += tr_NLL
                it_loss_sum += it_NLL


                tr_optimizer = self.tr_optimizer
                it_optimizer = self.it_optimizer

                # Optimizer step
                tr_optimizer.zero_grad()
                it_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tr_model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.it_model.parameters(), 1)
                tr_optimizer.step()
                it_optimizer.step()


            # epoch-level averages
            avg_tr    = tr_loss_sum / tr_batches if tr_batches else 0.0
            avg_it    = it_loss_sum / it_batches if it_batches else 0.0
            avg_total = avg_tr + avg_it

            train_loss_list.append(avg_total)
            print(f"[E:{epoch:02d}] train tr_loss={avg_tr:.4f}, it_loss={avg_it:.4f}, total={avg_total:.4f}")

            dev_acc, dev_f1 = self.evaluate(valid_loader)
            dev_acc_list.append(dev_acc)
            f1_scores.append(dev_f1)

            if dev_f1 > record_dev:
                record_dev = dev_f1
                torch.save(self.tr_model.state_dict(), f"./src/checkpoints/tr/{self.modelname}.pt")
                torch.save(self.it_model.state_dict(), f"./src/checkpoints/it/{self.modelname}.pt")
                patience = full_patience
            else:
                patience -= 1

        print("...Done!")
        return train_loss_list, dev_acc_list, f1_scores

    def evaluate(self, valid_loader: DataLoader):

        # put models to eval mode
        self.tr_model.eval()
        self.it_model.eval()

        # lists to hold predictions and labels
        all_predictions = []
        all_labels      = []
        all_tr_predictions = []
        all_it_predictions = []
        all_tr_labels = []
        all_it_labels = []

        with torch.no_grad():
            for words, labels, langs in tqdm(valid_loader, desc="Evaluating"):
                batch_size, seq_len = labels.shape
                device = labels.device
                hidden_size = self.tr_embedder.bert_model.config.hidden_size

                tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
                it_indices = (langs == 1).nonzero(as_tuple=True)[0]

                if len(tr_indices) > 0:
                    tr_sents   = [words[i] for i in tr_indices.cpu().numpy()]
                    tr_embeds  = self.tr_embedder.embed_sentences(tr_sents)
                    tr_embs    = pad_sequence(tr_embeds, batch_first=True, padding_value=0).to(device)
                    if tr_embs.size(1) < seq_len:
                        pad_amt = seq_len - tr_embs.size(1)
                        tr_embs = F.pad(tr_embs, (0, 0, 0, pad_amt), "constant", 0)
                else:
                    tr_embs = torch.zeros((0, seq_len, hidden_size), device=device)

                if len(it_indices) > 0:
                    it_sents  = [words[i] for i in it_indices.cpu().numpy()]
                    it_embeds = self.it_embedder.embed_sentences(it_sents)
                    it_embs   = pad_sequence(it_embeds, batch_first=True, padding_value=0).to(device)
                    if it_embs.size(1) < seq_len:
                        pad_amt = seq_len - it_embs.size(1)
                        it_embs = F.pad(it_embs, (0, 0, 0, pad_amt), "constant", 0)
                else:
                    it_embs = torch.zeros((0, seq_len, hidden_size), device=device)

                # forward passes, getting list-of-lists predictions
                tr_decode = self.tr_model(tr_embs, None)
                it_decode = self.it_model(it_embs, None)

                # turn those into (N_lang, seq_len) tensors
                tr_pred = self.decode_to_tensor(tr_decode, seq_len, device)
                it_pred = self.decode_to_tensor(it_decode, seq_len, device)

                # reassemble the full batch predictions (batch_size, seq_len) 
                full_pred = torch.full(
                    (batch_size, seq_len),
                    fill_value=0,
                    dtype=torch.long,
                    device=device
                )

                # get the full predicions while keeping original order
                full_pred[tr_indices] = tr_pred
                full_pred[it_indices] = it_pred

                # accumulate for global scores
                valid_mask   = labels.ne(0)  # ignore padding
                flat_mask = valid_mask.view(-1)
                flat_pred = full_pred.view(-1)[flat_mask]
                flat_lbl  = labels.view(-1)[flat_mask]
                all_predictions.extend(flat_pred.cpu().tolist())
                all_labels     .extend(flat_lbl.cpu().tolist())

                # 1) make masks for non-pad tokens
                tr_valid_mask = labels[tr_indices].ne(0)   # shape: (n_tr_sents, seq_len)
                it_valid_mask = labels[it_indices].ne(0)

                # 2) extract only the valid (non-zero) tokens
                tr_flat_pred  = tr_pred.masked_select(tr_valid_mask)    # 1D tensor of all TR predictions
                tr_flat_label = labels[tr_indices].masked_select(tr_valid_mask)

                it_flat_pred  = it_pred.masked_select(it_valid_mask)
                it_flat_label = labels[it_indices].masked_select(it_valid_mask)

                # 3) extend your global lists with the flattened Python lists
                all_tr_predictions.extend(tr_flat_pred.cpu().tolist())
                all_tr_labels     .extend(tr_flat_label.cpu().tolist())

                all_it_predictions.extend(it_flat_pred.cpu().tolist())
                all_it_labels     .extend(it_flat_label.cpu().tolist())


            # --- compute overall Accuracy & F1 ---
            print("\n")
            full_accuracy = accuracy_score(all_labels, all_predictions)
            full_f1       = f1_score(all_labels, all_predictions,
                                    average='macro', zero_division=0)
            print(f"Full Accuracy: {full_accuracy:.4f}, Full F1: {full_f1:.4f}")
            print(classification_report(all_tr_labels, all_tr_predictions,zero_division=0,digits=4))
            print("\n")

            tr_accuracy = accuracy_score(all_tr_labels, all_tr_predictions)
            tr_f1       = f1_score(all_tr_labels, all_tr_predictions,
                                    average='macro', zero_division=0)
            print(f"TR Accuracy: {tr_accuracy:.4f}, TR F1: {tr_f1:.4f}")
            print(classification_report(all_labels, all_predictions,zero_division=0,digits=4))
            print("\n")

            it_accuracy = accuracy_score(all_it_labels, all_it_predictions)
            it_f1       = f1_score(all_it_labels, all_it_predictions,
                                    average='macro', zero_division=0)
            print(f"IT Accuracy: {it_accuracy:.4f}, IT F1: {it_f1:.4f}")
            print(classification_report(all_it_labels, all_it_predictions,zero_division=0,digits=4))
            print("\n")

            #
            # save_predictions_to_csv
            # self._save_predictions_to_csv(tr_indices, tr_sents, all_tr_predictions, all_tr_labels, "tr")
            # self._save_predictions_to_csv(it_indices, it_sents, all_it_predictions, all_it_labels, "it")
        # return the actual values you computed
            
        return full_accuracy, full_f1

    

    
    def _save_predictions_to_csv(self, indices, words, predictions, true_labels, languages):
        """
        Save predictions to a CSV file, ordered by original indices.
        
        Args:
            indices: Original indices of the words
            words: The original words
            predictions: Predicted labels for each word
            true_labels: True labels for each word
            languages: Language of each sample (tr or it)
        """
        
        # Create a list to store data for the DataFrame
        data = []
        
        # Since indices are already in order, no need to sort
        for idx, word, pred, true, lang in zip(indices, words, predictions, true_labels, languages):
            # Convert list of labels to string representation
            pred_str = ','.join(pred) if pred else ''
            true_str = ','.join(true) if true else ''
            
            data.append({
                'Index': idx,
                'Language': lang,
                'Text': word,
                'Predicted_Labels': pred_str,
                'True_Labels': true_str
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        
        # Create directory if it doesn't exist
        os.makedirs("./results", exist_ok=True)
        
        # Save to CSV
        csv_path = "./results/predictions.csv"
        df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")

    def decode_to_tensor(self, decode_out, seq_len, device):
        # 1) list of lists → list of 1D tensors
        token_tensors = [torch.tensor(seq, dtype=torch.long, device=device)
                        for seq in decode_out]
        # 2) hiç prediction yoksa boş tensor
        if not token_tensors:
            return torch.zeros((0, seq_len), dtype=torch.long, device=device)
        # 3) pad_sequence ile batch_first ve padding_value=-1
        padded = pad_sequence(token_tensors, batch_first=True, padding_value=-1)
        # 4) eğer hâlâ seq_len’den kısa ise sağa pad et
        if padded.size(1) < seq_len:
            pad_amt = seq_len - padded.size(1)
            padded = F.pad(padded, (0, pad_amt), value=-1)
        return padded
    
    def embed_and_pad(self, words, indices, embedder):
        if len(indices) == 0:
            return torch.zeros((0, seq_len, hidden_size), device=device)
        # gather sentences
        sents = [words[i] for i in indices.cpu().numpy()]
        embs = embedder.embed_sentences(sents)
        embs = pad_sequence(embs, batch_first=True, padding_value=0).to(device)
        # right-pad to seq_len if needed
        if embs.size(1) < seq_len:
            pad_amt = seq_len - embs.size(1)
            embs = F.pad(embs, (0, 0, 0, pad_amt))
        return embs

    