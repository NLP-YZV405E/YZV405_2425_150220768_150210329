# Multilingual Idiom Detection System

This project implements a multilingual idiom detection system using BERT-based models for Turkish and Italian languages. The system is designed to identify idiomatic expressions in text using sequence labeling techniques.


You can try out the model using our Google Colab notebook. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XkrbL7KSuQN04tZ_Qv2VtLeNezH73jIC?usp=sharing)


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

You can install the required packages using pip and the `requirements.txt` file:


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

## Datasets
 
The system supports multiple datasets for training and evaluation:
 
1. **ID10M Dataset**
   - Created using the ID10M corpus
   - Contains Turkish and Italian idiomatic expressions
   - Used for training and evaluation
 
2. **ITU Dataset**
   - Created using the ITU corpus
   - Contains Turkish idiomatic expressions
   - Used for training and evaluation
 
3. **PARSEME Dataset**
   - From the PARSEME shared task
   - Contains multilingual idiomatic expressions
   - Used for training and evaluation
 
4. **COMBINED Dataset**
   - Combination of ITU, ID10M, and PARSEME datasets
   - Largest dataset with diverse examples
   - Used for training robust models
 
5. **ITU_TRAIN_DEV Dataset**
   - Special version of ITU dataset
   - Combines training and development sets
   - Used for training with more data
 
Each dataset is stored in the `resources/` directory under its respective folder. The data is in TSV format with three columns:
- Token: The word or token
- Tag: The label (B-IDIOM, I-IDIOM, O)
- Language: The language of the token (tr/it)

## Model Architecture

The system uses a hybrid architecture combining:
- BERT models for language-specific embeddings
- Optional LSTM layers for sequence modeling
- Task-specific classification layers
- CRF layer for sequence labeling
- Focal Loss for handling class imbalance
- Regularization techniques (e.g., Dropout, L2 Weight Decay) to prevent overfitting

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

The system supports three main modes of operation: training, testing, and updating. Each mode serves a different purpose and has specific requirements.

1. **Training Mode**:
```bash
python src/main.py
```
When prompted:
- Select mode: "train"
- Choose dataset: "ID10M", "ITU", "PARSEME", "COMBINED", "ITU_TRAIN_DEV", "CUSTOM"
- Enter model name
- Choose whether to fine-tune BERT

This mode trains new models from scratch without loading any existing checkpoints.
To use a custom dataset:
- Place your `train.csv`, `dev.csv`, and `test.csv` files into the `./data/CUSTOM/` folder.
- `dev.csv` must contain labels (used for F1 score calculation), while `test.csv` should not contain labels (used for generating predictions).
- To train a model using custom data, all three files (`train.csv`, `dev.csv`, and `test.csv`) are required.
- To test the model with custom data, provide `dev.csv` (if `test_mode` is 'dev' for evaluation) or `test.csv` (if `test_mode` is 'test' for predictions).
- Select "CUSTOM" when prompted for the dataset.

2. **Update Mode**:
```bash
python src/main.py
```
When prompted:
- Select mode: "update"
- Choose dataset
- Select checkpoint to update for Turkish and Italian models
- Enter new model name
- Choose whether to fine-tune BERT

This mode loads existing checkpoints and continues training from those points. It's used when:
- Continuing training from a previous checkpoint
- Fine-tuning existing models

For both testing and updating modes, you can choose to use different checkpoints for Turkish and Italian models, or select "none" if you don't want to use a checkpoint for a particular language.

2. **Testing Mode**:
```bash
python src/main.py
```
When prompted:
- Select mode: "test"
- Choose dataset
- Choose testmode: "test" uses test.csv and returns only predictions. "dev" uses dev.csv and evaluates performance.
- Select checkpoint to use for Turkish and Italian models

This mode loads pre-trained models and performs inference without any training.

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

## Results Organization

All experiment results are saved in the `results/` directory. Each experiment gets its own folder with a naming convention that includes key information about the experiment:

```
results/
└── datasetname_trmodelname_itmodelname_lr_.../
    ├── prediction.csv          # Model predictions
    ├── scores.json            # Evaluation scores
    ├── results.pdf            # Detailed evaluation report
    ├── dev_acc.png           # Development accuracy plot
    ├── dev_f1.png            # Development F1 score plot
    ├── loss.png              # Training vs validation loss plot
    ├── tr_loss.png           # Turkish model loss plot
    ├── it_loss.png           # Italian model loss plot
    └── bert_weight_changes.png # BERT weight changes during fine-tuning
```

The folder name includes:
- Dataset name (e.g., ID10M, ITU, PARSEME, COMBINED)
- Turkish model name
- Italian model name
- Learning rate
- Other relevant parameters

For example: `itu_basetr_xxlit_0005_regulizer_focalloss_trainbertafter15_lowerlr_40epoch`

Note: Not all experiment folders will contain the complete set of files shown above. Some folders might have fewer files if the training run was interrupted or if certain evaluation steps were skipped. The presence of specific files depends on the training progress and configuration.

## Model Checkpoints

Checkpoints are saved in:
- `checkpoints/tr/` for Turkish models
- `checkpoints/it/` for Italian models

## Authors

Efe Can Kırbıyık \
Berke Kurt

