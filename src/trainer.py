import model
import math
from __init__ import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import math

class Trainer:
    def __init__(self,
                 tr_model: nn.Module,
                 it_model: nn.Module,
                 tr_optimizer,
                 it_optimizer,
                 labels_vocab: dict):
        
        self.tr_model     = tr_model
        self.it_model     = it_model
        self.tr_optimizer = tr_optimizer
        self.it_optimizer = it_optimizer
        self.labels_vocab = labels_vocab

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

        device = next(self.tr_model.parameters()).device
        print("\nTraining...")

        train_loss_list = []
        dev_loss_list = []
        f1_scores = []
        record_dev = 0.0
        full_patience = patience

        for epoch in range(1, epochs + 1):
            if patience <= 0:
                print("Stopping early (no more patience).")
                break

            print(f" Epoch {epoch:03d}")

            self.tr_model.train()
            self.it_model.train()

            tr_loss = it_loss = total_loss = 0.0
            tr_batches = it_batches = 0

            for words, labels, langs in tqdm(train_loader, desc=f"Epoch {epoch}"):
                # Move to device
                words  = words.to(device)
                labels = labels.to(device)
                langs  = langs.to(device)

                tr_mask = (langs == 0)
                it_mask = (langs == 1)

                loss = 0.0
                parts = 0

                if tr_mask.any():
                    w_tr = words[tr_mask]
                    y_tr = labels[tr_mask]
                    tr_LL, _ = self.tr_model(w_tr, y_tr)
                    tr_NLL = -tr_LL.sum() / y_tr.size(0)
                    tr_loss += tr_NLL.item()
                    loss += tr_NLL
                    parts += 1
                    tr_batches += 1

                if it_mask.any():
                    w_it = words[it_mask]
                    y_it = labels[it_mask]
                    it_LL, _ = self.it_model(w_it, y_it)
                    it_NLL = -it_LL.sum() / y_it.size(0)
                    it_loss += it_NLL.item()
                    loss += it_NLL
                    parts += 1
                    it_batches += 1

                # Backpropagate combined loss
                loss = loss / parts
                self.tr_optimizer.zero_grad()
                self.it_optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.tr_model.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.it_model.parameters(), 1)

                self.tr_optimizer.step()
                self.it_optimizer.step()

                total_loss += loss.item()

            avg_tr = tr_loss / tr_batches if tr_batches > 0 else 0.0
            avg_it = it_loss / it_batches if it_batches > 0 else 0.0
            avg_total = total_loss / (tr_batches + it_batches) if (tr_batches + it_batches) > 0 else 0.0

            train_loss_list.append(avg_total)
            print(f"[E:{epoch:02d}] train loss = {avg_total:.4f}, tr_loss = {avg_tr:.4f}, it_loss = {avg_it:.4f}")

            valid_loss, f1 = self.evaluate(valid_loader)
            dev_loss_list.append(valid_loss)
            f1_scores.append(f1)

            if f1 > record_dev:
                record_dev = f1
                torch.save({
                    "tr_model": self.tr_model.state_dict(),
                    "it_model": self.it_model.state_dict(),
                }, f"./src/checkpoints/{modelname}.pt")
                patience = full_patience
            else:
                patience -= 1

            print(f"\t[E:{epoch:02d}] valid_loss={valid_loss:.4f}  f1={f1:.4f}  patience={patience}")

        print("...Done!")
        return train_loss_list, dev_loss_list, f1_scores

    def evaluate(self, valid_loader: DataLoader):
        device = next(self.tr_model.parameters()).device

        self.tr_model.eval()
        self.it_model.eval()

        tr_loss_sum = it_loss_sum = 0.0
        tr_batches = it_batches = 0

        all_predictions = []
        all_labels = []
        labels_vocab_reverse = {v: k for k, v in self.labels_vocab.items()}

        with torch.no_grad():
            for words, labels, langs in tqdm(valid_loader, desc="Evaluating"):
                words  = words.to(device)
                labels = labels.to(device)
                langs  = langs.to(device)

                tr_mask = (langs == 0)
                it_mask = (langs == 1)

                if tr_mask.any():
                    w_tr = words[tr_mask]
                    y_tr = labels[tr_mask]
                    ll_tr, preds_tr = self.tr_model(w_tr, y_tr)
                    nll_tr = -ll_tr.sum() / y_tr.size(0)
                    tr_loss_sum += nll_tr.item()
                    tr_batches += 1

                    preds_flat  = preds_tr.view(-1, preds_tr.size(-1))
                    labels_flat = y_tr.view(-1)
                    for i in range(len(preds_flat)):
                        if labels_flat[i] != 0:
                            pred_id = int(torch.argmax(preds_flat[i]))
                            true_id = int(labels_flat[i])
                            all_predictions.append(labels_vocab_reverse[pred_id])
                            all_labels.append(labels_vocab_reverse[true_id])

                if it_mask.any():
                    w_it = words[it_mask]
                    y_it = labels[it_mask]
                    ll_it, preds_it = self.it_model(w_it, y_it)
                    nll_it = -ll_it.sum() / y_it.size(0)
                    it_loss_sum += nll_it.item()
                    it_batches += 1

                    preds_flat  = preds_it.view(-1, preds_it.size(-1))
                    labels_flat = y_it.view(-1)
                    for i in range(len(preds_flat)):
                        if labels_flat[i] != 0:
                            pred_id = int(torch.argmax(preds_flat[i]))
                            true_id = int(labels_flat[i])
                            all_predictions.append(labels_vocab_reverse[pred_id])
                            all_labels.append(labels_vocab_reverse[true_id])

        avg_tr_loss = tr_loss_sum / tr_batches if tr_batches else 0.0
        avg_it_loss = it_loss_sum / it_batches if it_batches else 0.0
        total_batches = tr_batches + it_batches
        avg_total_loss = (tr_loss_sum + it_loss_sum) / total_batches if total_batches else 0.0

        print(f"[EVAL] tr_loss={avg_tr_loss:.4f} ({tr_batches} batches), it_loss={avg_it_loss:.4f} ({it_batches} batches), total_loss={avg_total_loss:.4f}")
        print(classification_report(all_labels, all_predictions, digits=3))
        f1 = f1_score(all_labels, all_predictions, average='macro')
        print(f"Macro F1: {f1:.4f}")

        return avg_total_loss, f1
