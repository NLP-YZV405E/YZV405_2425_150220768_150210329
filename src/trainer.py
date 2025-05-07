import model
import math
from __init__ import *

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import pandas as pd
import os
import numpy as np

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
        self.best_predictions = None

    def padding_mask(self, batch: torch.Tensor) -> torch.BoolTensor:
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        return padding.type(torch.bool)

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              epochs: int = 1,
              patience: int = 10,
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

            tr_loss_sum = it_loss_sum = 0.0
            tr_batches  = it_batches  = 0

            for words, labels, langs in tqdm(train_loader, desc=f"Epoch {epoch}"):
                labels = labels.to(self.device)
                langs  = langs.to(self.device)

                # get embeddings and pad to [B, L, H]
                tr_embs = pad_sequence(
                    self.tr_embedder.embed_sentences(words),
                    batch_first=True, padding_value=0
                ).to(self.device)
                it_embs = pad_sequence(
                    self.it_embedder.embed_sentences(words),
                    batch_first=True, padding_value=0
                ).to(self.device)

                # create language masks [B, L]
                tr_mask = (langs == 0)
                it_mask = (langs == 1)

                # zero out opposite-language labels
                tr_labels = labels.clone()
                it_labels = labels.clone()
                tr_labels[it_mask] = 0
                it_labels[tr_mask] = 0

                # forward passes
                tr_LL, _ = self.tr_model(tr_embs, tr_labels)
                it_LL, _ = self.it_model(it_embs, it_labels)

                # compute losses with zero-division guard
                if tr_mask.any().item():
                    tr_NLL = -tr_LL.sum() / tr_mask.sum().float()
                    tr_loss_sum += tr_NLL.item()
                    tr_batches  += 1
                else:
                    tr_NLL = torch.tensor(0.0, device=self.device)

                if it_mask.any().item():
                    it_NLL = -it_LL.sum() / it_mask.sum().float()
                    it_loss_sum += it_NLL.item()
                    it_batches  += 1
                else:
                    it_NLL = torch.tensor(0.0, device=self.device)

                loss = tr_NLL + it_NLL

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

            valid_loss, f1, predictions = self.evaluate(valid_loader)
            dev_loss_list.append(valid_loss)
            f1_scores.append(f1)

            if f1 > record_dev:
                record_dev = f1
                torch.save(self.tr_model.state_dict(), f"./src/checkpoints/tr/{modelname}.pt")
                torch.save(self.it_model.state_dict(), f"./src/checkpoints/it/{modelname}.pt")
                patience = full_patience
                # Save the best predictions
                self.best_predictions = predictions
                self.save_predictions_as_csv(f"./src/predictions/{modelname}_predictions.csv")
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
        
        # For prediction CSV
        sentence_indices = []
        sentence_languages = []
        prediction_indices = []
        sentence_id = 0
        
        with torch.no_grad():
            for words, labels, langs in tqdm(valid_loader, desc="Evaluating"):
                labels = labels.to(self.device)
                langs  = langs.to(self.device)

                tr_embs = pad_sequence(
                    self.tr_embedder.embed_sentences(words),
                    batch_first=True, padding_value=0
                ).to(self.device)
                it_embs = pad_sequence(
                    self.it_embedder.embed_sentences(words),
                    batch_first=True, padding_value=0
                ).to(self.device)

                tr_mask = (langs == 0)
                it_mask = (langs == 1)

                tr_labels = labels.clone()
                it_labels = labels.clone()
                tr_labels[it_mask] = 0
                it_labels[tr_mask] = 0

                ll_tr, preds_tr = self.tr_model(tr_embs, tr_labels)
                ll_it, preds_it = self.it_model(it_embs, it_labels)

                if tr_mask.any().item():
                    nll_tr = -ll_tr.sum() / tr_mask.sum().float()
                    tr_loss_sum += nll_tr.item()
                    tr_batches  += 1
                if it_mask.any().item():
                    nll_it = -ll_it.sum() / it_mask.sum().float()
                    it_loss_sum += nll_it.item()
                    it_batches  += 1

                # For each sentence in the batch
                for b in range(langs.size(0)):
                    lang_id = "tr" if langs[b, 0].item() == 0 else "it"
                    idiom_indices = []

                    # Choose which predictions to use based on language
                    if lang_id == "tr":
                        pred_tensor = preds_tr[b]
                        label_tensor = tr_labels[b]
                    else:
                        pred_tensor = preds_it[b]
                        label_tensor = it_labels[b]

                    # Find contiguous spans of idioms
                    current_span = []
                    for i in range(len(pred_tensor)):
                        # Skip padding
                        if label_tensor[i] == 0:
                            continue
                        
                        # Get predicted tag
                        pred_idx = torch.argmax(pred_tensor[i]).item()
                        pred_tag = labels_vocab_reverse[pred_idx]
                        
                        # Track idiom spans
                        if pred_tag == "B-IDIOM":
                            if current_span:
                                idiom_indices.append(current_span)
                                current_span = []
                            current_span = [i]
                        elif pred_tag == "I-IDIOM" and current_span:
                            current_span.append(i)
                        elif pred_tag == "O":
                            if current_span:
                                idiom_indices.append(current_span)
                                current_span = []
                    
                    # Don't forget the last span
                    if current_span:
                        idiom_indices.append(current_span)
                    
                    # If no idioms found, use [-1] as a marker
                    if not idiom_indices:
                        idiom_indices = [[-1]]
                    
                    # Store for CSV
                    sentence_indices.append(sentence_id)
                    sentence_languages.append(lang_id)
                    prediction_indices.append(str(idiom_indices))
                    sentence_id += 1

                # collect predictions for metrics
                preds_flat = preds_tr.view(-1, preds_tr.size(-1))
                labels_flat = tr_labels.view(-1)
                for i in range(len(preds_flat)):
                    if labels_flat[i] != 0:
                        all_predictions.append(labels_vocab_reverse[int(torch.argmax(preds_flat[i]))])
                        all_labels.append(labels_vocab_reverse[int(labels_flat[i])])
                preds_flat = preds_it.view(-1, preds_it.size(-1))
                labels_flat = it_labels.view(-1)
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

        # Create predictions dataframe
        predictions_data = {
            'id': sentence_indices,
            'indices': prediction_indices,
            'language': sentence_languages
        }

        return avg_total, f1, predictions_data
    
    def save_predictions_as_csv(self, output_path):
        """Save the predictions as a CSV file."""
        if self.best_predictions is None:
            print("No predictions to save.")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(self.best_predictions)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Also save a copy in the standard prediction.csv format
        standard_path = os.path.join(os.path.dirname(output_path), "prediction.csv")
        df.to_csv(standard_path, index=False)
        print(f"Standard prediction file saved as {standard_path}")
    
    def predict_and_save(self, test_loader, output_path="./src/predictions/prediction.csv"):
        """Run prediction on test data and save directly to CSV."""
        _, _, predictions = self.evaluate(test_loader)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
        print(f"Test predictions saved to {output_path}")
        return df
    