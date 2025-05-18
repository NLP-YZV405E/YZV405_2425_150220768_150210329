# Multilingual Idiom Detection System

This project implements a multilingual idiom detection system using BERT-based models for Turkish and Italian languages. The system is designed to identify idiomatic expressions in text using sequence labeling techniques.

## Features

- Multilingual support for Turkish and Italian
- BERT-based models with task-specific layers
- Conditional Random Field (CRF) for sequence labeling
- Focal Loss implementation for handling class imbalance
- Gradual BERT fine-tuning capabilities
- Comprehensive evaluation metrics and visualization
- Mixed precision training support

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm

## Project Structure

```
.
├── src/                    # Source code directory
│   ├── model.py           # Core model architecture
│   ├── dataset.py         # Dataset handling
│   ├── trainer.py         # Training and evaluation logic
│   ├── main.py           # Main execution script
│   ├── bert_embedder.py  # BERT embedding utilities
│   ├── collate.py        # Data collation functions
│   ├── hparams.py        # Hyperparameter definitions
│   ├── utils.py          # Utility functions
│   ├── scoring.py        # Scoring program implementation
│   ├── __init__.py       # Package initialization
│   └── checkpoints      # Model checkpoints directory
│       ├── tr          # Turkish model checkpoints
│       └── it          # Italian model checkpoints
├── resources            # Dataset directory
│   ├── ID10M
│   ├── ITU
│   ├── PARSEME
│   └── COMBINED
├── results             # Training results and visualizations
├── data               # Data directory
├── scoring_program    # Evaluation scoring program
├── papers            # Research papers and documentation
├── proposal          # Project proposal documents
```

## Model Architecture

The system uses a hybrid architecture combining:
- BERT models for language-specific embeddings
- Optional LSTM layers for sequence modeling
- Task-specific classification layers
- CRF layer for sequence labeling
- Focal Loss for handling class imbalance

## Hyperparameters Configuration

The model's hyperparameters can be configured by modifying the parameters in `src/hparams.py`. The available parameters include:

### Model Architecture Parameters
- `dropout`: Dropout rate (default: 0.5)
- `num_classes`: Number of output classes (default: 4)
- `bidirectional`: Whether to use bidirectional LSTM (default: True)
- `num_layers`: Number of LSTM layers (default: 3)
- `use_lstm`: Whether to use LSTM layers (default: True)
- `use_attention`: Whether to use attention mechanism (default: False)
- `device`: Device to run the model on (default: "cuda" if available, else "cpu")

### Training Parameters
- `batch_size`: Batch size for training (default: 32)
- `lr`: Learning rate (default: 0.001)
- `epoch`: Number of training epochs (default: 50)
- `warmup_steps`: Number of warmup steps (default: 1000)
- `weight_decay`: Weight decay for regularization (default: 0.001)
- `gradient_clip`: Gradient clipping value (default: 1)
- `scheduler_factor`: Learning rate scheduler factor (default: 0.5)
- `scheduler_patience`: Learning rate scheduler patience (default: 3)
- `focal_loss_weight`: Weight for focal loss component (default: 0.3)

## Training Process

The training process consists of two phases:

1. **Task-Specific Training**:
   - BERT layers are frozen
   - Only task-specific layers are trained
   - Uses standard cross-entropy loss

2. **BERT Fine-tuning** (Optional):
   - Gradual unfreezing of BERT layers
   - Three sub-phases:
     - Top layers fine-tuning
     - Middle layers fine-tuning
     - Full model fine-tuning
   - Uses reduced learning rates for BERT layers

## Usage

1. **Training**:
```bash
python src/main.py
```
When prompted:
- Select mode: "train"
- Choose dataset: "ID10M", "ITU", "PARSEME", "COMBINED", or "ITU_TRAIN_DEV"
- Enter model name
- Choose whether to fine-tune BERT

2. **Testing**:
```bash
python src/main.py
```
When prompted:
- Select mode: "test"
- Choose dataset
- Select checkpoint to use

3. **Model Update**:
```bash
python src/main.py
```
When prompted:
- Select mode: "update"
- Choose dataset
- Select checkpoint to update
- Enter new model name
- Choose whether to fine-tune BERT

## Evaluation

The system provides comprehensive evaluation metrics:
- Accuracy scores for each language
- F1 scores for each language
- Confusion matrices
- Training/validation loss curves
- BERT weight change tracking (during fine-tuning)

Results are saved in the `results/` directory, including:
- Prediction files
- Evaluation metrics
- Visualization plots
- Detailed classification reports

## Model Checkpoints

Checkpoints are saved in:
- `checkpoints/tr/` for Turkish models
- `checkpoints/it/` for Italian models

## Authors

Efe Can Kırbıyık \
Berke Kurt

## Acknowledgments

- BERT models from Hugging Face Transformers
- Turkish BERT: dbmdz/bert-base-turkish-128k-cased
- Italian BERT: dbmdz/bert-base-italian-xxl-cased 
