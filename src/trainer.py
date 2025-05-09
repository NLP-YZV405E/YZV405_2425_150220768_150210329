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
                 labels_vocab: dict):
        self.tr_model     = tr_model
        self.it_model     = it_model
        self.tr_optimizer = tr_optimizer
        self.it_optimizer = it_optimizer
        self.labels_vocab = labels_vocab
        self.tr_embedder  = tr_embedder
        self.it_embedder  = it_embedder
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"

    def padding_mask(self, batch: torch.Tensor) -> torch.BoolTensor:
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        return padding.type(torch.bool)

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              epochs: int = 1,
              patience: int = 20,
              modelname: str = "idiom_expr_detector"):

        print("\nTraining...")
        train_loss_list = []
        dev_loss_list   = []
        f1_scores       = []
        record_dev      = 0.0
        full_patience   = patience

        for epoch in range(1, epochs + 1):
            if patience <= 0:
                print("Stopping early (no more patience).")
                break

            print(f" Epoch {epoch:03d}")
            self.tr_model.train()
            self.it_model.train()

            # epochtaki loss ve batch sayıları
            tr_loss_sum = it_loss_sum = 0.0
            tr_batches  = it_batches  = 0

            for words, labels, langs in tqdm(train_loader, desc=f"Epoch {epoch}"):
                labels = labels.to(self.device)
                langs = langs.to(self.device)

                # len(words) = 16, labels -> [16, 14], langs -> [16, 14]
                # words -> list of list, words: [['Zaman', 'kazanmak', 'için', 'yaptığın', 'entrikalar', 'seni',
                # 'kurtarmayacak', ',', 'eninde', 'sonunda', 'yakalayacak', 'seni', 'polis', '!'], ...]
                # langs -> [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1], dil + pad, bunu değiştirelim
                # labels -> [ 3,  3,  3,  3,  3,  1,  2,  3, -1, -1, -1, -1, -1, -1], idiom + pad, bu iyi

                # .nonzero() ile zero olmayan indexleri alıyoruz, maskede kullanınca o dile ait indexler geliyor.
                tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
                it_indices = (langs == 1).nonzero(as_tuple=True)[0]
                
                # loss değerlerini 0 la
                tr_loss = torch.tensor(0.0, device=self.device)
                it_loss = torch.tensor(0.0, device=self.device)
                
                # if there are turkish data in the batch
                if len(tr_indices) > 0:
                    
                    tr_words = [words[i] for i in tr_indices.cpu().numpy()]
                    tr_labels_subset = labels[tr_indices]
                    # first the first batch tr_words.size = 

                    
                    # tr_embedded -> [batch_size, seq_len, hidden_size]
                    tr_embedded = self.tr_embedder.embed_sentences(tr_words)
                    print(f"tr_embedded shape: {len(tr_embedded)}")

                    # tr_embs_shape = [batch_size, seq_len, hidden_size] -> 62, 14, 768
                    # get embeddings for Turkish data

                    tr_embs = pad_sequence(
                        tr_embedded, batch_first=True, padding_value=-1
                    ).to(self.device)

                    print(f"tr_embs shape: {tr_embs.shape}")

                    
                    # forward pass and loss calculation
                    tr_LL, _ = self.tr_model(tr_embs, tr_labels_subset)
                    tr_NLL = -tr_LL.sum() / len(tr_indices)
                    tr_loss_sum += tr_NLL.item()
                    tr_batches += 1
                    tr_loss = tr_NLL
                
                # if there are italian data in the batch
                if len(it_indices) > 0:
                    it_words = [words[i] for i in it_indices.cpu().numpy()]
                    it_labels_subset = labels[it_indices]
                    
                    # get embeddings for italian data
                    it_embs = pad_sequence(
                        self.it_embedder.embed_sentences(it_words),
                        batch_first=True, padding_value=0
                    ).to(self.device)
                    
                    # forward pass and loss calculation
                    it_LL, _ = self.it_model(it_embs, it_labels_subset)
                    it_NLL = -it_LL.sum() / len(it_indices)
                    it_loss_sum += it_NLL.item()
                    it_batches += 1
                    it_loss = it_NLL
                
                # combine the losses
                loss = tr_loss + it_loss
                
                # Optimizer step
                self.tr_optimizer.zero_grad()
                self.it_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tr_model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.it_model.parameters(), 1)
                self.tr_optimizer.step()
                self.it_optimizer.step()

            # epoch-level averages
            avg_tr    = tr_loss_sum / tr_batches if tr_batches else 0.0
            avg_it    = it_loss_sum / it_batches if it_batches else 0.0
            avg_total = avg_tr + avg_it

            train_loss_list.append(avg_total)
            print(f"[E:{epoch:02d}] train tr_loss={avg_tr:.4f}, it_loss={avg_it:.4f}, total={avg_total:.4f}")

            valid_loss, f1 = self.evaluate(valid_loader)
            dev_loss_list.append(valid_loss)
            f1_scores.append(f1)

            if f1 > record_dev:
                record_dev = f1
                torch.save(self.tr_model.state_dict(), f"./src/checkpoints/tr/{modelname}.pt")
                torch.save(self.it_model.state_dict(), f"./src/checkpoints/it/{modelname}.pt")
                patience = full_patience
            else:
                patience -= 1

            print(f"\t[E:{epoch:02d}] valid_loss={valid_loss:.4f}  f1={f1:.4f}  patience={patience}")

        print("...Done!")
        return train_loss_list, dev_loss_list, f1_scores

    def evaluate(self, valid_loader: DataLoader):
        self.tr_model.eval()
        self.it_model.eval()

        tr_loss_sum = it_loss_sum = 0.0
        tr_batches  = it_batches  = 0
        all_predictions = []
        all_labels      = []
        labels_vocab_reverse = {v: k for k, v in self.labels_vocab.items()}

        with torch.no_grad():
            for words, labels, langs in tqdm(valid_loader, desc="Evaluating"):
                labels = labels.to(self.device)
                langs  = langs.to(self.device)

                # Split data by language while keeping original indices
                tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
                it_indices = (langs == 1).nonzero(as_tuple=True)[0]
                
                # Process Turkish data if available
                if len(tr_indices) > 0:
                    # Select Turkish data
                    tr_words = [words[i] for i in tr_indices.cpu().numpy()]
                    tr_labels_subset = labels[tr_indices]
                    
                    # Get embeddings for Turkish data
                    tr_embs = pad_sequence(
                        self.tr_embedder.embed_sentences(tr_words),
                        batch_first=True, padding_value=0
                    ).to(self.device)
                    
                    # Forward pass for Turkish model
                    ll_tr, preds_tr = self.tr_model(tr_embs, tr_labels_subset)
                    nll_tr = -ll_tr.sum() / len(tr_indices)
                    tr_loss_sum += nll_tr.item()
                    tr_batches += 1
                    
                    # Collect predictions for Turkish data
                    preds_flat = preds_tr.view(-1, preds_tr.size(-1))
                    labels_flat = tr_labels_subset.view(-1)
                    for i in range(len(preds_flat)):
                        if labels_flat[i] != 0:
                            all_predictions.append(labels_vocab_reverse[int(torch.argmax(preds_flat[i]))])
                            all_labels.append(labels_vocab_reverse[int(labels_flat[i])])
                
                # Process Italian data if available
                if len(it_indices) > 0:
                    # Select Italian data
                    it_words = [words[i] for i in it_indices.cpu().numpy()]
                    it_labels_subset = labels[it_indices]
                    
                    # Get embeddings for Italian data
                    it_embs = pad_sequence(
                        self.it_embedder.embed_sentences(it_words),
                        batch_first=True, padding_value=0
                    ).to(self.device)
                    
                    # Forward pass for Italian model
                    ll_it, preds_it = self.it_model(it_embs, it_labels_subset)
                    nll_it = -ll_it.sum() / len(it_indices)
                    it_loss_sum += nll_it.item()
                    it_batches += 1
                    
                    # Collect predictions for Italian data
                    preds_flat = preds_it.view(-1, preds_it.size(-1))
                    labels_flat = it_labels_subset.view(-1)
                    for i in range(len(preds_flat)):
                        if labels_flat[i] != 0:
                            all_predictions.append(labels_vocab_reverse[int(torch.argmax(preds_flat[i]))])
                            all_labels.append(labels_vocab_reverse[int(labels_flat[i])])

        avg_tr    = tr_loss_sum / tr_batches if tr_batches else 0.0
        avg_it    = it_loss_sum / it_batches if it_batches else 0.0
        avg_total = avg_tr + avg_it

        print(f"[EVAL] tr_loss={avg_tr:.4f}, it_loss={avg_it:.4f}, total_loss={avg_total:.4f}")
        print(classification_report(all_labels, all_predictions, digits=3, zero_division=0))
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        print(f"Macro F1: {f1:.4f}")

        return avg_total, f1
    
    def test(self, valid_loader: DataLoader):
        self.tr_model.eval()
        self.it_model.eval()

        tr_loss_sum = it_loss_sum = 0.0
        tr_batches  = it_batches  = 0
        
        # Lists to store results in order
        ordered_predictions = []
        ordered_labels = []
        ordered_words = []
        ordered_langs = []
        sample_indices = []
        
        # For metrics calculation
        all_predictions = []
        all_labels = []
        
        labels_vocab_reverse = {v: k for k, v in self.labels_vocab.items()}
        
        # Sample index counter
        sample_idx = 0

        with torch.no_grad():
            for words, labels, langs in tqdm(valid_loader, desc="Testing"):
                # Since batch size is 1, we can simplify by taking the first item
                word = words[0]  # Single sentence 
                label = labels.to(self.device)  # Shape: [1, seq_len]
                lang = langs.to(self.device)[0]  # Single language indicator
                
                # Determine language (0=Turkish, 1=Italian)
                is_turkish = (lang.item() == 0)
                
                if is_turkish:
                    # Process Turkish sample
                    tr_embs = pad_sequence(
                        self.tr_embedder.embed_sentences([word]),
                        batch_first=True, padding_value=0
                    ).to(self.device)
                    
                    # Forward pass
                    ll_tr, preds_tr = self.tr_model(tr_embs, label)
                    nll_tr = -ll_tr.sum()
                    tr_loss_sum += nll_tr.item()
                    tr_batches += 1
                    
                    # Collect predictions
                    preds = []
                    true_labels = []
                    
                    # Process each token
                    for j in range(preds_tr.size(1)):
                        if j < label.size(1) and label[0, j] != 0:
                            pred_label = labels_vocab_reverse[int(torch.argmax(preds_tr[0, j]))]
                            true_label = labels_vocab_reverse[int(label[0, j])]
                            
                            preds.append(pred_label)
                            true_labels.append(true_label)
                            
                            # Add to flat lists for metrics
                            all_predictions.append(pred_label)
                            all_labels.append(true_label)
                else:
                    # Process Italian sample
                    it_embs = pad_sequence(
                        self.it_embedder.embed_sentences([word]),
                        batch_first=True, padding_value=0
                    ).to(self.device)
                    
                    # Forward pass
                    ll_it, preds_it = self.it_model(it_embs, label)
                    nll_it = -ll_it.sum()
                    it_loss_sum += nll_it.item()
                    it_batches += 1
                    
                    # Collect predictions
                    preds = []
                    true_labels = []
                    
                    # Process each token
                    for j in range(preds_it.size(1)):
                        if j < label.size(1) and label[0, j] != 0:
                            pred_label = labels_vocab_reverse[int(torch.argmax(preds_it[0, j]))]
                            true_label = labels_vocab_reverse[int(label[0, j])]
                            
                            preds.append(pred_label)
                            true_labels.append(true_label)
                            
                            # Add to flat lists for metrics
                            all_predictions.append(pred_label)
                            all_labels.append(true_label)
                
                # Store results in order
                ordered_predictions.append(preds)
                ordered_labels.append(true_labels)
                ordered_words.append(word)
                ordered_langs.append("tr" if is_turkish else "it")
                sample_indices.append(sample_idx)
                
                # Increment sample counter
                sample_idx += 1

        # Calculate metrics
        avg_tr = tr_loss_sum / tr_batches if tr_batches else 0.0
        avg_it = it_loss_sum / it_batches if it_batches else 0.0
        avg_total = avg_tr + avg_it

        print(f"[TEST] tr_loss={avg_tr:.4f}, it_loss={avg_it:.4f}, total_loss={avg_total:.4f}")
        print(classification_report(all_labels, all_predictions, digits=3, zero_division=0))
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        print(f"Macro F1: {f1:.4f}")
        
        # Save predictions to CSV
        self._save_predictions_to_csv(sample_indices, ordered_words, ordered_predictions, ordered_labels, ordered_langs)

        return all_labels, all_predictions
    
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