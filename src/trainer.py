from __init__ import *
from hparams import HParams

class Trainer:
    def __init__(self,
                 tr_model: nn.Module,
                 it_model: nn.Module,
                 tr_optimizer,
                 it_optimizer,
                 labels_vocab: dict,
                 modelname = "idiom_expr_detector",
                 train_bert = False):
        
        self.tr_model = tr_model
        self.it_model = it_model
        self.tr_optimizer = tr_optimizer
        self.it_optimizer = it_optimizer
        self.labels_vocab = labels_vocab
        self.modelname = modelname
        self.tr_hidden = tr_model.bert_embedder.bert_model.config.hidden_size
        self.it_hidden = it_model.bert_embedder.bert_model.config.hidden_size
        self.params = HParams()
        self.result_dir = f"./results/{modelname}/"
        os.makedirs(self.result_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Flag for whether to train BERT layers after task layers
        self.train_bert = train_bert
        
        # Initialize mixed precision training with increased stability
        self.scaler = torch.amp.GradScaler(enabled=True, device=self.device, 
                                        init_scale=2**10,
                                        growth_factor=1.5,  
                                        backoff_factor=0.5, 
                                        growth_interval=100) 
        
        # Create bert optimizers for bert training
        if self.train_bert:
            # Lower learning rate for fintuning BERT
            tr_bert_lr = 5e-6  
            it_bert_lr = 5e-6 

            # Create separate optimizer groups for different BERT layers to enable gradual unfreezing
            tr_bert_params = []
            
            # Create non-overlapping parameter groups for TR model
            # Group 1: Top layers (layers -4 to -1)
            top_layer_params = []
            for i in range(-4, 0):
                layer_params = list(self.tr_model.bert_embedder.bert_model.encoder.layer[i].parameters())
                top_layer_params.extend(layer_params)
            tr_bert_params.append({
                'params': top_layer_params,
                'lr': tr_bert_lr
            })
            
            # Group 2: Middle layers (layers -8 to -5)
            middle_layer_params = []
            for i in range(-8, -4):
                layer_params = list(self.tr_model.bert_embedder.bert_model.encoder.layer[i].parameters())
                middle_layer_params.extend(layer_params)
            tr_bert_params.append({
                'params': middle_layer_params,
                'lr': tr_bert_lr
            })
            
            # Group 3: everything else
            # First get all parameters
            all_params = set(self.tr_model.bert_embedder.bert_model.parameters())
            # Remove parameters already in groups 1 and 2
            grouped_params = set(top_layer_params + middle_layer_params)
            remaining_params = list(all_params - grouped_params)
            
            tr_bert_params.append({
                'params': remaining_params,
                'lr': tr_bert_lr
            })
            
            # Similar structure for Italian
            it_bert_params = []
            
            # Group 1: Top layers (layers -4 to -1)
            it_top_layer_params = []
            for i in range(-4, 0):
                layer_params = list(self.it_model.bert_embedder.bert_model.encoder.layer[i].parameters())
                it_top_layer_params.extend(layer_params)
            it_bert_params.append({
                'params': it_top_layer_params,
                'lr': it_bert_lr
            })
            
            # Group 2: Middle layers (layers -8 to -5)
            it_middle_layer_params = []
            for i in range(-8, -4):
                layer_params = list(self.it_model.bert_embedder.bert_model.encoder.layer[i].parameters())
                it_middle_layer_params.extend(layer_params)
            it_bert_params.append({
                'params': it_middle_layer_params,
                'lr': it_bert_lr * 0.75
            })
            
            # Group 3: everything else
            
            it_all_params = set(self.it_model.bert_embedder.bert_model.parameters())
            it_grouped_params = set(it_top_layer_params + it_middle_layer_params)
            it_remaining_params = list(it_all_params - it_grouped_params)
            
            it_bert_params.append({
                'params': it_remaining_params,
                'lr': it_bert_lr * 0.5
            })
            
            self.tr_bert_optimizer = torch.optim.AdamW(
                tr_bert_params,
                weight_decay=0.0005,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            self.it_bert_optimizer = torch.optim.AdamW(
                it_bert_params,
                weight_decay=0.0005,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Initialize flags for gradual unfreezing
            self.bert_unfreeze_phase = 0  # 0: None unfrozen, 1: Top layers, 2: Middle, 3: All
        
        # Initialize learning rate schedulers
        self.tr_scheduler = ReduceLROnPlateau(
            self.tr_optimizer,
            mode='max',
            factor=self.params.scheduler_factor,
            patience=self.params.scheduler_patience
        )
        self.it_scheduler = ReduceLROnPlateau(
            self.it_optimizer,
            mode='max',
            factor=self.params.scheduler_factor,
            patience=self.params.scheduler_patience
        )
        
        if self.train_bert:
            self.tr_bert_scheduler = ReduceLROnPlateau(
                self.tr_bert_optimizer,
                mode='max',
                factor=self.params.scheduler_factor,
                patience=self.params.scheduler_patience,
                verbose=True
            )
            
            self.it_bert_scheduler = ReduceLROnPlateau(
                self.it_bert_optimizer,
                mode='max',
                factor=self.params.scheduler_factor,
                patience=self.params.scheduler_patience,
                verbose=True
            )

        # Verify BERT frozen status
        is_bert_frozen = not any(p.requires_grad for p in tr_model.bert_embedder.bert_model.parameters())
        print("BERT layers are initially frozen:", is_bert_frozen)

    def padding_mask(self, batch: torch.Tensor) -> torch.BoolTensor:
        padding = torch.ones_like(batch)
        padding[batch == 0] = 0
        return padding.type(torch.bool)

    def train(self,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              epochs: int = 40,
              patience: int = 25):

        print("\nTraining...")
        # Lists to save metrics
        train_loss_list = []
        tr_train_loss_list = []
        it_train_loss_list = []
        dev_acc_list    = []
        f1_scores       = []
        dev_tr_loss_list = []
        dev_it_loss_list = []
        record_dev      = 0.0
        
        bert_weight_changes = []
        
        if self.train_bert:
            print("=== PHASE 1: Training task-specific layers (BERT frozen) ===")
            total_epochs = epochs
            phase1_epochs = 15  # 15 epochs for task layers
            phase2_epochs = total_epochs - phase1_epochs  # Remaining for BERT fine-tuning
            
            print(f"Task-specific training: {phase1_epochs} epochs")
            print(f"BERT fine-tuning: {phase2_epochs} epochs")
            
            # Store best models from phase 1
            best_tr_state_dict = None
            best_it_state_dict = None
            
            # Phase 1: Train with frozen BERT
            full_patience = patience
            current_patience = full_patience
            
            for epoch in range(1, phase1_epochs + 1):
                if current_patience <= 0:
                    print("Stopping Phase 1 early (no more patience).")
                    break
                    
                print(f" Epoch {epoch:03d}/{phase1_epochs}, patience: {current_patience}")
                
                # Ensure BERT is frozen
                self.tr_model.freeze_bert()
                self.it_model.freeze_bert()
                
                # Train for one epoch
                tr_loss, it_loss = self._train_epoch(epoch, train_loader)
                
                tr_train_loss_list.append(tr_loss)
                it_train_loss_list.append(it_loss)
                train_loss_list.append(tr_loss + it_loss)
                
                # Evaluate on validation set
                dev_acc, dev_f1, dev_tr_loss, dev_it_loss = self.evaluate(valid_loader, record_dev)
                dev_acc_list.append(dev_acc)
                f1_scores.append(dev_f1)
                dev_tr_loss_list.append(dev_tr_loss)
                dev_it_loss_list.append(dev_it_loss)
                
                # Update learning rates
                self.tr_scheduler.step(dev_f1)
                self.it_scheduler.step(dev_f1)
                
                # Check if we have a new best model
                if dev_f1 > record_dev:
                    record_dev = dev_f1
                    best_tr_state_dict = copy.deepcopy(self.tr_model.state_dict())
                    best_it_state_dict = copy.deepcopy(self.it_model.state_dict())
                    current_patience = full_patience
                    print(f"New best model! F1: {dev_f1:.4f}")
                else:
                    current_patience -= 1
            
            # Load best model from Phase 1
            if best_tr_state_dict is not None:
                self.tr_model.load_state_dict(best_tr_state_dict)
                self.it_model.load_state_dict(best_it_state_dict)
                print("Loaded best model from Phase 1 for BERT fine-tuning")
            
            # Save task-specific model before BERT fine-tuning
            torch.save(best_tr_state_dict, f"./src/checkpoints/tr/{self.modelname}_task_only.pt")
            torch.save(best_it_state_dict, f"./src/checkpoints/it/{self.modelname}_task_only.pt")
            print(f"Saved task-only models to {self.modelname}_task_only.pt")
            
            # Reset patience for Phase 2
            current_patience = full_patience
            phase1_record_dev = record_dev
            
            # Phase 2: Gradual fine-tuning of BERT layers
            print("\n=== PHASE 2: Gradually fine-tuning BERT layers ===")
            
            # Reset best performance for phase 2
            record_dev = phase1_record_dev  
            current_patience = full_patience
            
            # Divide Phase 2 into 3 sub-phases for gradual unfreezing
            epochs_per_unfreeze = max(phase2_epochs // 5, 1)
            
            # Phase 2a: Unfreeze only the top layers
            self.bert_unfreeze_phase = 1
            print("\n=== Phase 2a: Fine-tuning top BERT layers only ===")
            
            # Freeze all BERT layers first
            for param in self.tr_model.bert_embedder.bert_model.parameters():
                param.requires_grad = False
            for param in self.it_model.bert_embedder.bert_model.parameters():
                param.requires_grad = False
                
            # Unfreeze only the top 4 layers
            for i in range(-4, 0):
                for param in self.tr_model.bert_embedder.bert_model.encoder.layer[i].parameters():
                    param.requires_grad = True
                for param in self.it_model.bert_embedder.bert_model.encoder.layer[i].parameters():
                    param.requires_grad = True
                    
            print("Unfrozen top 4 BERT layers for both models")
            
            # Train for the first sub-phase
            for epoch in range(1, epochs_per_unfreeze + 1):
                if current_patience <= 0:
                    print("Stopping Phase 2a early (no more patience).")
                    break
                
                print(f" BERT top layers fine-tuning epoch {epoch:03d}/{epochs_per_unfreeze}, patience: {current_patience}")
                
                # Train with top BERT layers unfrozen
                tr_loss, it_loss = self._train_epoch_with_bert(epoch, train_loader)
                
                tr_train_loss_list.append(tr_loss)
                it_train_loss_list.append(it_loss)
                train_loss_list.append(tr_loss + it_loss)
                
                # Evaluate to get dev accuracy and F1 score
                dev_acc, dev_f1, dev_tr_loss, dev_it_loss = self.evaluate(valid_loader, record_dev)
                dev_acc_list.append(dev_acc)
                f1_scores.append(dev_f1)
                dev_tr_loss_list.append(dev_tr_loss)
                dev_it_loss_list.append(dev_it_loss)
                
                # Update learning rates
                self.tr_scheduler.step(dev_f1)
                self.it_scheduler.step(dev_f1)
                self.tr_bert_scheduler.step(dev_f1)
                self.it_bert_scheduler.step(dev_f1)
                
                # Check for new best model
                if dev_f1 > record_dev:
                    record_dev = dev_f1
                    best_tr_state_dict = copy.deepcopy(self.tr_model.state_dict())
                    best_it_state_dict = copy.deepcopy(self.it_model.state_dict())
                    current_patience = full_patience
                    print(f"New best model with top BERT layers fine-tuned! F1: {dev_f1:.4f}")
                else:
                    current_patience -= 1
            
            # Load best model from Phase 2a before proceeding
            if best_tr_state_dict is not None:
                self.tr_model.load_state_dict(best_tr_state_dict)
                self.it_model.load_state_dict(best_it_state_dict)
                print("Loaded best model from Phase 2a")
                
            # Reset patience
            current_patience = full_patience
            
            # Phase 2b: Unfreeze middle layers too
            self.bert_unfreeze_phase = 2
            print("\n=== Phase 2b: Fine-tuning middle BERT layers ===")
            
            # Unfreeze middle 4 layers (layers 4-8 from top)
            for i in range(-8, -4):
                for param in self.tr_model.bert_embedder.bert_model.encoder.layer[i].parameters():
                    param.requires_grad = True
                for param in self.it_model.bert_embedder.bert_model.encoder.layer[i].parameters():
                    param.requires_grad = True
                    
            print("Unfrozen middle 4 BERT layers for both models")
            
            # Train for the second sub-phase
            for epoch in range(1, epochs_per_unfreeze + 1):
                if current_patience <= 0:
                    print("Stopping Phase 2b early (no more patience).")
                    break
                
                print(f" BERT middle layers fine-tuning epoch {epoch:03d}/{epochs_per_unfreeze}, patience: {current_patience}")
                
                # Train with middle BERT layers unfrozen
                tr_loss, it_loss = self._train_epoch_with_bert(epoch, train_loader)
                
                tr_train_loss_list.append(tr_loss)
                it_train_loss_list.append(it_loss)
                train_loss_list.append(tr_loss + it_loss)
                
                # Evaluate
                dev_acc, dev_f1, dev_tr_loss, dev_it_loss = self.evaluate(valid_loader, record_dev)
                dev_acc_list.append(dev_acc)
                f1_scores.append(dev_f1)
                dev_tr_loss_list.append(dev_tr_loss)
                dev_it_loss_list.append(dev_it_loss)
                
                # Update learning rates
                self.tr_scheduler.step(dev_f1)
                self.it_scheduler.step(dev_f1)
                self.tr_bert_scheduler.step(dev_f1)
                self.it_bert_scheduler.step(dev_f1)
                
                # Check for new best model
                if dev_f1 > record_dev:
                    record_dev = dev_f1
                    best_tr_state_dict = copy.deepcopy(self.tr_model.state_dict())
                    best_it_state_dict = copy.deepcopy(self.it_model.state_dict())
                    current_patience = full_patience
                    print(f"New best model with middle BERT layers fine-tuned! F1: {dev_f1:.4f}")
                else:
                    current_patience -= 1
            
            # Load best model from Phase 2b before proceeding
            if best_tr_state_dict is not None:
                self.tr_model.load_state_dict(best_tr_state_dict)
                self.it_model.load_state_dict(best_it_state_dict)
                print("Loaded best model from Phase 2b")
                
            # Reset patience
            current_patience = full_patience
            
            # Phase 2c: Unfreeze all layers (but with lowest learning rate for Turkish early layers)
            self.bert_unfreeze_phase = 3
            print("\n=== Phase 2c: Fine-tuning all BERT layers ===")
            
            # Unfreeze remaining layers
            self.tr_model.unfreeze_bert()
            self.it_model.unfreeze_bert()
            
            # Train for the final sub-phase
            remaining_epochs = phase2_epochs - 2 * epochs_per_unfreeze
            
            for epoch in range(1, remaining_epochs + 1):
                if current_patience <= 0:
                    print("Stopping Phase 2c early (no more patience).")
                    break
                
                print(f" BERT all layers fine-tuning epoch {epoch:03d}/{remaining_epochs}, patience: {current_patience}")
                
                # Track BERT weight changes
                initial_q = (self.tr_model
                    .bert_embedder
                    .bert_model
                    .encoder
                    .layer[0]
                    .attention
                    .self
                    .query
                    .weight
                    .detach()
                    .cpu()
                    .clone()
                )
                
                # Train with all BERT layers unfrozen
                tr_loss, it_loss = self._train_epoch_with_bert(epoch, train_loader)
                
                tr_train_loss_list.append(tr_loss)
                it_train_loss_list.append(it_loss)
                train_loss_list.append(tr_loss + it_loss)
                
                # Evaluate
                dev_acc, dev_f1, dev_tr_loss, dev_it_loss = self.evaluate(valid_loader, record_dev)
                dev_acc_list.append(dev_acc)
                f1_scores.append(dev_f1)
                dev_tr_loss_list.append(dev_tr_loss)
                dev_it_loss_list.append(dev_it_loss)
                
                # Update learning rates
                self.tr_scheduler.step(dev_f1)
                self.it_scheduler.step(dev_f1)
                self.tr_bert_scheduler.step(dev_f1)
                self.it_bert_scheduler.step(dev_f1)
                
                # Calculate BERT weight changes
                final_q = (
                    self.tr_model.bert_embedder
                        .bert_model
                        .encoder
                        .layer[0]
                        .attention
                        .self
                        .query
                        .weight
                        .detach()
                        .cpu()
                )
                diff = final_q - initial_q
                change_norm = diff.norm().item()
                bert_weight_changes.append(change_norm)
                
                print(f"BERT Weight ΔL2 norm: {change_norm:.6f}")
                
                # Check for new best model
                if dev_f1 > record_dev:
                    record_dev = dev_f1
                    best_tr_state_dict = copy.deepcopy(self.tr_model.state_dict())
                    best_it_state_dict = copy.deepcopy(self.it_model.state_dict())
                    current_patience = full_patience
                    print(f"New best model with all BERT layers fine-tuned! F1: {dev_f1:.4f}")
                else:
                    current_patience -= 1
            
            # Load best model from Phase 2c
            if best_tr_state_dict is not None:
                self.tr_model.load_state_dict(best_tr_state_dict)
                self.it_model.load_state_dict(best_it_state_dict)
            
        else:
            # Single phase training - only task-specific layers
            print("=== Training only task-specific layers (BERT frozen) ===")
            
            full_patience = patience
            current_patience = full_patience
            
            best_tr_state_dict = None
            best_it_state_dict = None
            
            for epoch in range(1, epochs + 1):
                if current_patience <= 0:
                    print("Stopping early (no more patience).")
                    break
                
                print(f" Epoch {epoch:03d}/{epochs}, patience: {current_patience}")
                
                # Ensure BERT is frozen
                self.tr_model.freeze_bert()
                self.it_model.freeze_bert()
                
                # Train for current epoch
                tr_loss, it_loss = self._train_epoch(epoch, train_loader)
                
                tr_train_loss_list.append(tr_loss)
                it_train_loss_list.append(it_loss)
                train_loss_list.append(tr_loss + it_loss)
                
                # Evaluate
                dev_acc, dev_f1, dev_tr_loss, dev_it_loss = self.evaluate(valid_loader, record_dev)
                dev_acc_list.append(dev_acc)
                f1_scores.append(dev_f1)
                dev_tr_loss_list.append(dev_tr_loss)
                dev_it_loss_list.append(dev_it_loss)
                
                # Update learning rates
                self.tr_scheduler.step(dev_f1)
                self.it_scheduler.step(dev_f1)
                
                # Check for new best model
                if dev_f1 > record_dev:
                    record_dev = dev_f1
                    best_tr_state_dict = copy.deepcopy(self.tr_model.state_dict())
                    best_it_state_dict = copy.deepcopy(self.it_model.state_dict())
                    current_patience = full_patience
                    print(f"New best model! F1: {dev_f1:.4f}")
                else:
                    current_patience -= 1
            
            # Load best model
            if best_tr_state_dict is not None:
                self.tr_model.load_state_dict(best_tr_state_dict)
                self.it_model.load_state_dict(best_it_state_dict)
        
        # Save final models
        torch.save(self.tr_model.state_dict(), f"./src/checkpoints/tr/{self.modelname}.pt")
        torch.save(self.it_model.state_dict(), f"./src/checkpoints/it/{self.modelname}.pt")
        
        # Convert tensor metrics to float lists
        dev_acc_list      = self.to_float_list(dev_acc_list)
        f1_scores         = self.to_float_list(f1_scores)
        train_loss_list   = self.to_float_list(train_loss_list)
        tr_train_loss_list= self.to_float_list(tr_train_loss_list)
        it_train_loss_list= self.to_float_list(it_train_loss_list)
        dev_tr_loss_list  = self.to_float_list(dev_tr_loss_list)
        dev_it_loss_list  = self.to_float_list(dev_it_loss_list)
        
        # Plot BERT weight changes if we did BERT fine-tuning
        if self.train_bert and bert_weight_changes:
            plt.figure(figsize=(15, 5))
            sns.lineplot(x=list(range(1, len(bert_weight_changes) + 1)), y=bert_weight_changes)
            plt.title("BERT Weight Changes During Fine-tuning")
            plt.xlabel("Epoch")
            plt.ylabel("L2 Norm of Weight Change")
            plt.xticks(range(1, len(bert_weight_changes) + 1))
            plt.grid()
            plt.savefig(f"{self.result_dir}/bert_weight_changes.png")
            plt.close()

        # Compute combined dev loss
        dev_loss_list = [t + i for t, i in zip(dev_tr_loss_list, dev_it_loss_list)]

        # Plot metrics with their max markers
        # Plot accuracy
        self.plot_with_max(
            x=None, y=dev_acc_list,
            title="Dev Accuracy", ylabel="Accuracy",
            fname="dev_acc.png"
        )

        # Plot F1 scores
        self.plot_with_max(
            x=None, y=f1_scores,
            title="Dev F1 Score", ylabel="F1 Score",
            fname="dev_f1.png"
        )

        # Plot train vs dev loss
        self.plot_with_max(
            x=None, y=train_loss_list,
            title="Train vs Dev Loss", ylabel="Loss",
            fname="loss.png",
        )

        # Plot combined train vs dev loss
        epochs = list(range(1, len(train_loss_list) + 1))
        plt.figure(figsize=(25, 5))
        sns.lineplot(x=epochs, y=train_loss_list, label="Train Loss")
        sns.lineplot(x=epochs, y=dev_loss_list, label="Dev Loss")
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

        # Plot TR train vs dev loss
        self.plot_with_max(
            x=None, y=tr_train_loss_list,
            title="Train vs Dev Loss (TR)", ylabel="Loss",
            fname="tr_loss.png", label="Train Loss"
        )

        plt.figure(figsize=(25, 5))
        sns.lineplot(x=list(range(1, len(tr_train_loss_list) + 1)),
                    y=tr_train_loss_list, label="Train Loss")
        sns.lineplot(x=list(range(1, len(dev_tr_loss_list) + 1)),
                    y=dev_tr_loss_list, label="Dev Loss")
        # Mark min dev_tr_loss
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

        # Plot IT train vs dev loss
        self.plot_with_max(
            x=None, y=it_train_loss_list,
            title="Train vs Dev Loss (IT)", ylabel="Loss",
            fname="it_loss.png", label="Train Loss"
        )
        
        # Overlay dev_it_loss
        plt.figure(figsize=(25, 5))
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

        print("...Training complete!")
        return train_loss_list, dev_acc_list, f1_scores
    
    def _train_epoch(self, epoch, train_loader):
        """Train for one epoch with BERT frozen"""
        self.tr_model.train()
        self.it_model.train()
        
        # Ensure BERT is in eval mode (frozen)
        self.tr_model.bert_embedder.bert_model.eval()
        self.it_model.bert_embedder.bert_model.eval()
        
        tr_loss_sum = it_loss_sum = 0
        tr_batches = it_batches = 0
        
        for words, labels, langs in tqdm(train_loader, desc=f"Epoch {epoch}"):
            
            batch_size, seq_len = labels.shape
            
            # Split indices by language
            tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
            it_indices = (langs == 1).nonzero(as_tuple=True)[0]

            tr_NLL = torch.tensor(0., device=self.device)
            it_NLL = torch.tensor(0., device=self.device)

            if len(tr_indices) > 0:
                tr_batches += 1
                tr_labels = labels[tr_indices]
                tr_sents  = [words[i] for i in tr_indices.cpu().numpy()]
                with torch.amp.autocast(device_type=self.device):
                    tr_NLL, _ = self.tr_model(tr_sents, tr_labels, seq_len)

            if len(it_indices) > 0:
                it_batches += 1
                it_labels = labels[it_indices]
                it_sents  = [words[i] for i in it_indices.cpu().numpy()]
                with torch.amp.autocast(device_type=self.device):
                    it_NLL, _ = self.it_model(it_sents, it_labels, seq_len)

            # Combined loss
            loss = tr_NLL + it_NLL
            tr_loss_sum += tr_NLL.item()
            it_loss_sum += it_NLL.item()

            # Backward the combined loss
            self.scaler.scale(loss).backward()

            # Only unscale optimizers that saw gradients
            if len(tr_indices) > 0:
                self.scaler.unscale_(self.tr_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.tr_model.parameters(),
                    self.params.gradient_clip
                )
                self.scaler.step(self.tr_optimizer)

            if len(it_indices) > 0:
                self.scaler.unscale_(self.it_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.it_model.parameters(),
                    self.params.gradient_clip
                )
                self.scaler.step(self.it_optimizer)

            # update the scaler for next iteration
            # it reduces the scale if loss is inf or nan in the previous iteration
            self.scaler.update()

            # zero out both gradients
            self.tr_optimizer.zero_grad()
            self.it_optimizer.zero_grad()
        
        # Calculate epoch-level average losses
        avg_tr = tr_loss_sum / tr_batches if tr_batches else 0.0
        avg_it = it_loss_sum / it_batches if it_batches else 0.0
        
        print(f"[Epoch {epoch}] Train TR loss: {avg_tr:.4f}, IT loss: {avg_it:.4f}, Total: {avg_tr + avg_it:.4f}")
        
        return avg_tr, avg_it
    
    def _train_epoch_with_bert(self, epoch, train_loader):
        """Train for one epoch with BERT unfrozen"""
        self.tr_model.train()
        self.it_model.train()
        
        # Ensure BERT is in train mode
        self.tr_model.bert_embedder.bert_model.train()
        self.it_model.bert_embedder.bert_model.train()
        
        tr_loss_sum = it_loss_sum = 0
        tr_batches = it_batches = 0
        
        for words, labels, langs in tqdm(train_loader, desc=f"BERT fine-tuning epoch {epoch}"):
            batch_size, seq_len = labels.shape
            
            tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
            it_indices = (langs == 1).nonzero(as_tuple=True)[0]
            tr_NLL = torch.tensor(0., device=self.device)
            it_NLL = torch.tensor(0., device=self.device)

            if len(tr_indices) > 0:
                tr_batches += 1
                tr_labels = labels[tr_indices]
                tr_sents  = [words[i] for i in tr_indices.cpu().numpy()]
                with torch.amp.autocast(device_type=self.device):
                    tr_NLL, _ = self.tr_model(tr_sents, tr_labels, seq_len)

            if len(it_indices) > 0:
                it_batches += 1
                it_labels = labels[it_indices]
                it_sents  = [words[i] for i in it_indices.cpu().numpy()]
                with torch.amp.autocast(device_type=self.device):
                    it_NLL, _ = self.it_model(it_sents, it_labels, seq_len)

            loss = tr_NLL + it_NLL
            tr_loss_sum += tr_NLL.item()
            it_loss_sum += it_NLL.item()

            # Zero all grads (model + BERT)
            self.tr_optimizer.zero_grad()
            self.it_optimizer.zero_grad()
            self.tr_bert_optimizer.zero_grad()
            self.it_bert_optimizer.zero_grad()

            self.scaler.scale(loss).backward()

            # Choose clip_value by phase
            if self.bert_unfreeze_phase == 1:
                clip_value = 1.0
            elif self.bert_unfreeze_phase == 2:
                clip_value = 0.5
            else:
                clip_value = 0.1

            # Turkish optimizers
            if len(tr_indices) > 0:
                self.scaler.unscale_(self.tr_optimizer)
                self.scaler.unscale_(self.tr_bert_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.tr_model.parameters(),
                    clip_value
                )
                self.scaler.step(self.tr_optimizer)
                self.scaler.step(self.tr_bert_optimizer)

            # Italian optimizers
            if len(it_indices) > 0:
                self.scaler.unscale_(self.it_optimizer)
                self.scaler.unscale_(self.it_bert_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.it_model.parameters(),
                    clip_value
                )
                self.scaler.step(self.it_optimizer)
                self.scaler.step(self.it_bert_optimizer)

            # Scale for next iteration
            self.scaler.update()
        
        # Calculate epoch-level average losses
        avg_tr = tr_loss_sum / tr_batches if tr_batches else 0.0
        avg_it = it_loss_sum / it_batches if it_batches else 0.0
        
        print(f"[BERT fine-tuning epoch {epoch}] Train TR loss: {avg_tr:.4f}, IT loss: {avg_it:.4f}, Total: {avg_tr + avg_it:.4f}")
        
        return avg_tr, avg_it

    def evaluate(self, valid_loader: DataLoader, record_dev):

        # Put models to eval mode
        self.tr_model.eval()
        self.it_model.eval()
        
        self.tr_model.bert_embedder.bert_model.eval()
        self.it_model.bert_embedder.bert_model.eval()
        # Lists to hold predictions and labels
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

        # For prediction.csv
        csv_rows   = []
        global_idx = 0                      

        with torch.no_grad():
            for words, labels, langs in tqdm(valid_loader, desc="Evaluating"):
                batch_size, seq_len = labels.shape
                device = labels.device

                tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
                it_indices = (langs == 1).nonzero(as_tuple=True)[0]

                tr_loss = it_loss = 0

                if len(tr_indices) > 0:
                    tr_batches += 1
                    tr_sents   = [words[i] for i in tr_indices.cpu().numpy()]
                    tr_decode = self.tr_model(tr_sents, None, seq_len)
                    # Skip loss calculation for external evaluation to avoid label mismatch errors
                    if record_dev > -1:  # This is during training
                        tr_loss,_ = self.tr_model(tr_sents, labels[tr_indices], seq_len)
                    else:
                        tr_loss = 0
                
                else: 
                    tr_decode = []
                    tr_loss = 0

                if len(it_indices) > 0:
                    it_batches += 1
                    it_sents  = [words[i] for i in it_indices.cpu().numpy()]
                    it_decode = self.it_model(it_sents, None, seq_len)
                    # Skip loss calculation for external evaluation to avoid label mismatch errors
                    if record_dev > -1:  # This is during training
                        it_loss,_ = self.it_model(it_sents, labels[it_indices],seq_len)
                    else:
                        it_loss = 0
                
                else:
                    it_decode = []
                    it_loss = 0

                # Forward passes, getting list-of-lists predictions
                tr_loss_sum += tr_loss
                it_loss_sum += it_loss

                # Turn those into (N_lang, seq_len) tensors
                tr_pred = self.decode_to_tensor(tr_decode, seq_len, device)
                it_pred = self.decode_to_tensor(it_decode, seq_len, device)

                # Reassemble the full batch predictions (batch_size, seq_len) 
                full_pred = torch.full(
                    (batch_size, seq_len),
                    fill_value=0,
                    dtype=torch.long,
                    device=device
                )

                # Get the full predicions while keeping original order
                full_pred[tr_indices] = tr_pred
                full_pred[it_indices] = it_pred

                # For prediction.csv
                # Collect per‑sentence CSV rows
                for row_idx in range(batch_size):
                    # Mask to ignore padding
                    valid_tok_mask = labels[row_idx].ne(0)
                    sent_pred      = full_pred[row_idx][valid_tok_mask]
                    # Indices where prediction is NOT label 0
                    if self.params.num_classes == 3:
                        not_token_labels = [0, 2]
                    else:
                        not_token_labels = [0, 3]
                    pred_indices   = [i for i, lbl in enumerate(sent_pred.tolist()) if lbl not in not_token_labels]
                    if not pred_indices:
                        pred_indices = [-1]
                    # Determine language
                    lang_str = "tr" if langs[row_idx].item() == 0 else "it"
                    # Append row
                    csv_rows.append({
                        "id": global_idx,
                        "indices": json.dumps(pred_indices),
                        "language": lang_str
                    })
                    global_idx += 1

                # Accumulate for global scores
                valid_mask   = labels.ne(0)  # ignore padding
                flat_mask = valid_mask.view(-1)
                flat_pred = full_pred.view(-1)[flat_mask]
                flat_lbl  = labels.view(-1)[flat_mask]
                all_predictions.extend(flat_pred.cpu().tolist())
                all_labels     .extend(flat_lbl.cpu().tolist())

                tr_valid_mask = labels[tr_indices].ne(0)   # shape: (n_tr_sents, seq_len)
                it_valid_mask = labels[it_indices].ne(0)

                tr_flat_pred  = tr_pred.masked_select(tr_valid_mask)
                tr_flat_label = labels[tr_indices].masked_select(tr_valid_mask)

                it_flat_pred  = it_pred.masked_select(it_valid_mask)
                it_flat_label = labels[it_indices].masked_select(it_valid_mask)

                all_tr_predictions.extend(tr_flat_pred.cpu().tolist())
                all_tr_labels     .extend(tr_flat_label.cpu().tolist())

                all_it_predictions.extend(it_flat_pred.cpu().tolist())
                all_it_labels     .extend(it_flat_label.cpu().tolist())

            tr_accuracy = accuracy_score(all_tr_labels, all_tr_predictions)
            it_accuracy = accuracy_score(all_it_labels, all_it_predictions)
            full_accuracy = accuracy_score(all_labels, all_predictions)

            pd.DataFrame(csv_rows).to_csv(f"{self.result_dir}/prediction.csv", index=False)
            scores = scoring_program(
                    truth_file=r"./data/public_data/eval.csv",
                    prediction_file=f"{self.result_dir}/prediction.csv",
                    score_output=f"{self.result_dir}/scores.json"
                )

            tr_f1 = scores["f1-score-tr"]
            it_f1 = scores["f1-score-it"]
            full_f1 = scores["f1-score-avg"]

            print(f"TR F1: {tr_f1:.4f}, IT F1: {it_f1:.4f}, Full F1: {full_f1:.4f}")

            if full_f1 > record_dev:

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
                        fig = plt.figure()
                        fig.set_size_inches(10,10)
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

        self.tr_model.eval()
        self.it_model.eval()

        self.tr_model.bert_embedder.bert_model.eval()
        self.it_model.bert_embedder.bert_model.eval()

        # For prediction.csv
        csv_rows   = []                      
        global_idx = 0                

        with torch.no_grad():
            for words, labels, langs in tqdm(test_loader, desc="Evaluating"):
                batch_size, seq_len = labels.shape
                device = labels.device

                tr_indices = (langs == 0).nonzero(as_tuple=True)[0]
                it_indices = (langs == 1).nonzero(as_tuple=True)[0]

                if len(tr_indices) > 0:
                    tr_sents   = [words[i] for i in tr_indices.cpu().numpy()]
                    tr_decode = self.tr_model(tr_sents, None, seq_len)

                else:
                    tr_decode = []


                if len(it_indices) > 0:
                    it_sents  = [words[i] for i in it_indices.cpu().numpy()]
                    it_decode = self.it_model(it_sents, None, seq_len)

                else:
                    it_decode = []    

                # Forward passes, getting list-of-lists predictions

                # Turn those into (N_lang, seq_len) tensors
                tr_pred = self.decode_to_tensor(tr_decode, seq_len, device)
                it_pred = self.decode_to_tensor(it_decode, seq_len, device)

                # Reassemble the full batch predictions (batch_size, seq_len) 
                full_pred = torch.full(
                    (batch_size, seq_len),
                    fill_value=0,
                    dtype=torch.long,
                    device=device
                )

                # Get the full predicions while keeping original order
                full_pred[tr_indices] = tr_pred
                full_pred[it_indices] = it_pred

                # Collect per‑sentence CSV rows for prediction.csv
                for row_idx in range(batch_size):
                    # Mask to ignore padding
                    valid_tok_mask = labels[row_idx].ne(0)
                    sent_pred      = full_pred[row_idx][valid_tok_mask]
                    # Indices where prediction is NOT label 0
                    if self.params.num_classes == 3:
                        not_token_labels = [0, 2]
                    else:
                        not_token_labels = [0, 3]
                    pred_indices   = [i for i, lbl in enumerate(sent_pred.tolist()) if lbl not in not_token_labels]
                    if not pred_indices:
                        pred_indices = [-1]
                    # Determine language str
                    lang_str = "tr" if langs[row_idx].item() == 0 else "it"
                    # Append row
                    csv_rows.append({
                        "id": global_idx,
                        "indices": json.dumps(pred_indices),
                        "language": lang_str
                    })
                    global_idx += 1

        # For prediction.csv
        save_csv_path = f"{self.result_dir}/test/prediction.csv"
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        pd.DataFrame(csv_rows).to_csv(save_csv_path, index=False)
        print(f"Predictions saved to {save_csv_path}")

    def decode_to_tensor(self, decode_out, seq_len, device):
        # list of lists → list of 1D tensors
        token_tensors = [torch.tensor(seq, dtype=torch.long, device=device)
                        for seq in decode_out]
        # hiç prediction yoksa boş tensor
        if not token_tensors:
            return torch.zeros((0, seq_len), dtype=torch.long, device=device)
        # pad_sequence ile batch_first ve padding_value=-1
        padded = pad_sequence(token_tensors, batch_first=True, padding_value=-1)
        # eğer hâlâ seq_len'den kısa ise sağa pad et
        if padded.size(1) < seq_len:
            pad_amt = seq_len - padded.size(1)
            padded = F.pad(padded, (0, pad_amt), value=-1)
        return padded
    
    def to_float_list(self, tensor_list):
            return [t.cpu().item() if hasattr(t, 'cpu') else float(t) for t in tensor_list]


    def plot_with_max(self, x, y, title, ylabel, fname, label=None):
        epochs = list(range(1, len(y) + 1))
        plt.figure(figsize=(25, 5))
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
    