import json
import os
import pandas as pd
from scoring import scoring_program

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
        self.result_dir   = f"./results/{modelname}/"
        os.makedirs(self.result_dir, exist_ok=True)
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"

    def padding_mask(self, batch: torch.Tensor) -> torch.BoolTensor:
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        return padding.type(torch.bool)

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              epochs: int = 50,
              patience: int = 10):

        print("\nTraining...")
        train_loss_list = []
        tr_train_loss_list = []
        it_train_loss_list = []
        dev_acc_list    = []
        f1_scores       = []
        dev_tr_loss_list = []
        dev_it_loss_list = []
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
            tr_loss_sum = it_loss_sum = 0
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

            print(f"[E:{epoch:02d}] train tr_loss={avg_tr:.4f}, it_loss={avg_it:.4f}, total={avg_total:.4f}")

            train_loss_list.append(avg_total)
            tr_train_loss_list.append(avg_tr)
            it_train_loss_list.append(avg_it)
            
            dev_acc, dev_f1, dev_tr_loss, dev_it_loss = self.evaluate(valid_loader, record_dev)
            dev_acc_list.append(dev_acc)
            f1_scores.append(dev_f1)
            dev_tr_loss_list.append(dev_tr_loss)
            dev_it_loss_list.append(dev_it_loss)

            if dev_f1 > record_dev:
                record_dev = dev_f1
                tr_state_dict = self.tr_model.state_dict()
                it_state_dict = self.it_model.state_dict()
                patience = full_patience

            else:
                patience -= 1


        # Convert all lists of tensors to lists of floats on CPU 
        dev_acc_list      = self.to_float_list(dev_acc_list)
        f1_scores         = self.to_float_list(f1_scores)
        train_loss_list   = self.to_float_list(train_loss_list)
        tr_train_loss_list= self.to_float_list(tr_train_loss_list)
        it_train_loss_list= self.to_float_list(it_train_loss_list)
        dev_tr_loss_list  = self.to_float_list(dev_tr_loss_list)
        dev_it_loss_list  = self.to_float_list(dev_it_loss_list)

        # compute combined dev loss
        dev_loss_list = [t + i for t, i in zip(dev_tr_loss_list, dev_it_loss_list)]


        #  Plot each metric with its max marker 
        self.plot_with_max(
            x=None, y=dev_acc_list,
            title="Dev Accuracy", ylabel="Accuracy",
            fname="dev_acc.png"
        )

        self.plot_with_max(
            x=None, y=f1_scores,
            title="Dev F1 Score", ylabel="F1 Score",
            fname="dev_f1.png"
        )

        self.plot_with_max(
            x=None, y=train_loss_list,
            title="Train vs Dev Loss", ylabel="Loss",
            fname="loss.png",
        )

        epochs = list(range(1, len(train_loss_list) + 1))
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=epochs, y=train_loss_list, label="Train Loss")
        sns.lineplot(x=epochs, y=dev_loss_list,    label="Dev Loss")
        min_idx = int(np.argmin(dev_loss_list)) + 1
        min_val = dev_loss_list[min_idx - 1]
        plt.axvline(min_idx, linestyle='--', color='green')
        plt.text(min_idx + 0.1, min_val, f"Epoch {min_idx}\n{min_val:.4f}",
                va='bottom', ha='left')
        plt.title("Train vs Dev Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(epochs)
        plt.grid()
        plt.legend()
        plt.savefig(f"{self.result_dir}/loss.png")
        plt.close()

        # And similarly for the TR / IT splits:
        self.plot_with_max(
            x=None, y=tr_train_loss_list,
            title="Train vs Dev Loss (TR)", ylabel="Loss",
            fname="tr_loss.png", label="Train Loss"
        )

        plt.figure(figsize=(10, 5))
        sns.lineplot(x=list(range(1, len(tr_train_loss_list) + 1)),
                    y=tr_train_loss_list, label="Train Loss")
        sns.lineplot(x=list(range(1, len(dev_tr_loss_list) + 1)),
                    y=dev_tr_loss_list, label="Dev Loss")
        # mark min dev_tr_loss
        min_idx = int(np.argmin(dev_tr_loss_list)) + 1
        min_val = dev_tr_loss_list[min_idx - 1]
        plt.axvline(min_idx, linestyle='--', color='green')
        plt.text(min_idx + 0.1, min_val, f"Epoch {min_idx}\n{min_val:.4f}",
                va='bottom', ha='left')
        plt.title("Train vs Dev Loss (TR)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(1, len(tr_train_loss_list) + 1))
        plt.grid()
        plt.legend()
        plt.savefig(f"{self.result_dir}/tr_loss.png")
        plt.close()

        self.plot_with_max(
            x=None, y=it_train_loss_list,
            title="Train vs Dev Loss (IT)", ylabel="Loss",
            fname="it_loss.png", label="Train Loss"
        )
        # overlay dev_it_loss
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=list(range(1, len(it_train_loss_list) + 1)),
                    y=it_train_loss_list, label="Train Loss")
        sns.lineplot(x=list(range(1, len(dev_it_loss_list) + 1)),
                    y=dev_it_loss_list, label="Dev Loss")
        min_idx = int(np.argmin(dev_it_loss_list)) + 1
        min_val = dev_it_loss_list[min_idx - 1]
        plt.axvline(min_idx, linestyle='--', color='green')
        plt.text(min_idx + 0.1, min_val, f"Epoch {min_idx}\n{min_val:.4f}",
                va='bottom', ha='left')
        plt.title("Train vs Dev Loss (IT)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(range(1, len(it_train_loss_list) + 1))
        plt.grid()
        plt.legend()
        plt.savefig(f"{self.result_dir}/it_loss.png")
        plt.close()

        torch.save(tr_state_dict, f"./src/checkpoints/tr/{self.modelname}.pt")
        torch.save(it_state_dict, f"./src/checkpoints/it/{self.modelname}.pt")

        print("...Done!")
        return train_loss_list, dev_acc_list, f1_scores

    def evaluate(self, valid_loader: DataLoader, record_dev):

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

        tr_loss_sum = 0
        it_loss_sum = 0
        tr_batches = 0
        it_batches = 0

        # for prediction.csv
        csv_rows   = []                      # rows that will become prediction.csv
        global_idx = 0                      # running sample id counter

        with torch.no_grad():
            for words, labels, langs in tqdm(valid_loader, desc="Evaluating"):
                batch_size, seq_len = labels.shape
                device = labels.device
                hidden_size = self.tr_embedder.bert_model.config.hidden_size

                tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
                it_indices = (langs == 1).nonzero(as_tuple=True)[0]

                if len(tr_indices) > 0:
                    tr_batches += 1
                    tr_sents   = [words[i] for i in tr_indices.cpu().numpy()]
                    tr_embeds  = self.tr_embedder.embed_sentences(tr_sents)
                    tr_embs    = pad_sequence(tr_embeds, batch_first=True, padding_value=0).to(device)
                    if tr_embs.size(1) < seq_len:
                        pad_amt = seq_len - tr_embs.size(1)
                        tr_embs = F.pad(tr_embs, (0, 0, 0, pad_amt), "constant", 0)
                else:
                    tr_embs = torch.zeros((0, seq_len, hidden_size), device=device)

                if len(it_indices) > 0:
                    it_batches += 1
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
                tr_loss,_ = self.tr_model(tr_embs, labels[tr_indices])
                it_loss,_ = self.it_model(it_embs, labels[it_indices])
                tr_loss_sum += -tr_loss
                it_loss_sum += -it_loss

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

                # for prediction.csv
                # collect per‑sentence CSV rows
                for row_idx in range(batch_size):
                    # mask to ignore padding
                    valid_tok_mask = labels[row_idx].ne(0)
                    sent_pred      = full_pred[row_idx][valid_tok_mask]
                    # indices where prediction is NOT label 0
                    pred_indices   = [i for i, lbl in enumerate(sent_pred.tolist()) if lbl not in [0, 3]]
                    if not pred_indices:
                        pred_indices = [-1]
                    # determine language str
                    lang_str = "tr" if langs[row_idx].item() == 0 else "it"
                    # append row
                    csv_rows.append({
                        "id": global_idx,
                        "indices": json.dumps(pred_indices),
                        "language": lang_str
                    })
                    global_idx += 1

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

            tr_f1 = f1_score(all_tr_labels, all_tr_predictions, average='macro', zero_division=0)
            it_f1 = f1_score(all_it_labels, all_it_predictions, average='macro', zero_division=0)
            full_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            tr_accuracy = accuracy_score(all_tr_labels, all_tr_predictions)
            it_accuracy = accuracy_score(all_it_labels, all_it_predictions)
            full_accuracy = accuracy_score(all_labels, all_predictions)

            print(f"TR F1: {tr_f1:.4f}, IT F1: {it_f1:.4f}, Full F1: {full_f1:.4f}")


            if full_f1 > record_dev:
                pd.DataFrame(csv_rows).to_csv(f"{self.result_dir}/predictions.csv", index=False)
                scores = scoring_program(
                    truth_file=r"./data/public_data/eval.csv",
                    prediction_file=f"{self.result_dir}/predictions.csv",
                    score_output=f"{self.result_dir}/scores.json"
                )

            with io.StringIO() as buffer:
                print(f"Scoring program: {scores}", file=buffer)
                print(f"Full Accuracy: {full_accuracy:.4f}, Full F1: {full_f1:.4f}", file=buffer)
                print(classification_report(all_labels, all_predictions,zero_division=0,digits=4), file=buffer)
                print("\n", file=buffer)
                print(f"TR Accuracy: {tr_accuracy:.4f}, TR F1: {tr_f1:.4f}", file=buffer)
                print(classification_report(all_tr_labels, all_tr_predictions,zero_division=0,digits=4), file=buffer)
                print("\n", file=buffer)
                print(f"IT Accuracy: {it_accuracy:.4f}, IT F1: {it_f1:.4f}", file=buffer)
                print(classification_report(all_it_labels, all_it_predictions,zero_division=0,digits=4), file=buffer)
                filename = f"{self.result_dir}/results.pdf"
                with PdfPages(filename, "w") as pdf:
                    # Page 1: Text Metrics
                    buffer.seek(0)
                    results_text = buffer.getvalue()
                    fig = plt.figure(figsize=(10, 10))
                    plt.axis('off')
                    plt.text(0, 1, results_text, verticalalignment='top', fontsize=10, fontfamily='monospace')
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                    # Page 2: Confusion Matrix
                    conf_matrix_full = confusion_matrix(all_labels, all_predictions)
                    conf_matrix_tr = confusion_matrix(all_tr_labels, all_tr_predictions)
                    conf_matrix_it = confusion_matrix(all_it_labels, all_it_predictions)
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].set_title("Confusion Matrix (full)")
                    axes[0].set_xlabel("Predicted Class")
                    axes[0].set_ylabel("True Class")

                    sns.heatmap(
                        conf_matrix_full, annot=True, fmt='d', cmap='Blues', ax=axes[0]
                    )
                    axes[1].set_title("Confusion Matrix (TR)")
                    axes[1].set_xlabel("Predicted Class")
                    axes[1].set_ylabel("True Class")

                    sns.heatmap(
                        conf_matrix_tr, annot=True, fmt='d', cmap='Blues',ax=axes[1]
                    )

                    axes[2].set_title("Confusion Matrix (IT)")
                    axes[2].set_xlabel("Predicted Class")
                    axes[2].set_ylabel("True Class")
                    sns.heatmap(
                        conf_matrix_it, annot=True, fmt='d', cmap='Blues', ax=axes[2]
                    )
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

        return full_accuracy, full_f1, tr_loss_sum / tr_batches, it_loss_sum / it_batches
    
    def test(self, test_loader: DataLoader):

        # put models to eval mode
        self.tr_model.eval()
        self.it_model.eval()

        # for prediction.csv
        csv_rows   = []                      
        global_idx = 0                

        with torch.no_grad():
            for words, labels, langs in tqdm(test_loader, desc="Evaluating"):
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

                # collect per‑sentence CSV rows for prediction.csv
                for row_idx in range(batch_size):
                    # mask to ignore padding
                    valid_tok_mask = labels[row_idx].ne(0)
                    sent_pred      = full_pred[row_idx][valid_tok_mask]
                    # indices where prediction is NOT label 0
                    pred_indices   = [i for i, lbl in enumerate(sent_pred.tolist()) if lbl not in [0, 3]]
                    if not pred_indices:
                        pred_indices = [-1]
                    # determine language str
                    lang_str = "tr" if langs[row_idx].item() == 0 else "it"
                    # append row
                    csv_rows.append({
                        "id": global_idx,
                        "indices": json.dumps(pred_indices),
                        "language": lang_str
                    })
                    global_idx += 1

        # for prediction.csv
        # >>>>> CHANGED START (actually write the CSV if a path is given)
        save_csv_path = f"{self.result_dir}/test/predictions.csv"
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(save_csv_path, index=False)
        print(f"Predictions saved to {save_csv_path}")

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
    
    def to_float_list(self, tensor_list):
            return [t.cpu().item() if hasattr(t, 'cpu') else float(t) for t in tensor_list]


    def plot_with_max(self, x, y, title, ylabel, fname, label=None):
        epochs = list(range(1, len(y) + 1))
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=epochs, y=y, label=label)
        # find max
        max_idx = int(np.argmax(y)) + 1
        max_val = y[max_idx - 1]
        plt.axvline(max_idx, linestyle='--', color='red')
        plt.text(max_idx + 0.1, max_val, f"Epoch {max_idx}\n{max_val:.4f}", 
                va='bottom', ha='left')
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.xticks(epochs)
        plt.grid()
        if label:
            plt.legend()
        plt.savefig(f"{self.result_dir}/{fname}")
        plt.close()
    