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

                # collect predictions
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

        return avg_total, f1
    
    # bunu düzeltelim şu an çalışmaz -> test buraya gelmeli predictions csvye kaydedilmeli
    def test(self, valid_loader: DataLoader):
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

                # collect predictions
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

        print(f"[TEST] tr_loss={avg_tr:.4f}, it_loss={avg_it:.4f}, total_loss={avg_total:.4f}")
        print(classification_report(all_labels, all_predictions, digits=3, zero_division=0))
        f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
        print(f"Macro F1: {f1:.4f}")

        return all_labels, all_predictions