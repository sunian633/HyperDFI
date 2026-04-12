# HyperDFI

HyperDFI is a PyTorch-based graph learning project for **drug-food interaction prediction**. The code builds a bipartite drug-food graph, learns drug and food representations with graph and hypergraph propagation, and predicts interaction labels with a classifier.

## Task

The current codebase is adapted for:

- Drug-food interaction prediction
- Bipartite graph representation learning

The dataset loader expects interaction triples in CSV format:

```text
drug_id,food_id,relation
```

## Project Structure

```text
HyperDFI/
├── Code/
│   ├── Main.py
│   ├── Model.py
│   ├── DataLoader.py
│   ├── Params.py
│   └── Utils/
│       ├── RunLogger.py
│       └── Utils.py
└── Data/
    └── DrugBank-df/
        └── transductive/
            ├── train.csv
            └── test.csv
```

## Core Files

- `Code/Main.py`: training and evaluation entry point
- `Code/Model.py`: model definition
- `Code/DataLoader.py`: dataset loading, ID remapping, adjacency construction
- `Code/Params.py`: command-line hyperparameters
- `Code/Utils/RunLogger.py`: simple runtime logger
- `Code/Utils/Utils.py`: loss and normalization helpers

## Model Overview

The model contains three main parts:

1. Embedding layer
   Initializes learnable embeddings for drugs and foods.
2. Graph propagation
   Uses a GCN-style sparse propagation layer on the bipartite adjacency matrix.
3. Hypergraph propagation
   Uses hypergraph-style transformations to capture higher-order structure.
4. Classifier
   Concatenates drug and food embeddings and predicts the interaction label.

The final embedding is the sum of propagated representations across layers.

## Data Processing

`Code/DataLoader.py` performs the following steps:

- Reads `train.csv` and `test.csv`
- Renames columns internally as `drug_nodes`, `food_nodes`, and `relations`
- Maps raw drug and food IDs to contiguous indices
- Builds the label matrix
- Creates sparse training and testing interaction matrices
- Converts the bipartite adjacency matrix to a CUDA sparse tensor
- Wraps samples into PyTorch `DataLoader` objects

## Environment

Recommended environment:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- SciPy
- scikit-learn
- wandb

You can install the common dependencies with:

```bash
pip install torch numpy pandas scipy scikit-learn wandb
```

## How To Run

Run from the `Code` directory:

```bash
cd Code
python Main.py
```

Example with explicit parameters:

```bash
python Main.py --data DrugBank-df --epoch 100 --batch 1024 --lr 1e-3 --gpu 0
```


## Training

The training script:

- loads the dataset
- initializes the model and optimizer
- trains for multiple epochs

Reported metrics include:

- Accuracy
- Precision
- F1
- AUC
- AUPR

## Notes

- The code currently uses CUDA tensors in several places, so GPU execution is expected by default.
- The dataset path is resolved relative to the `Code` directory.

## Citation

If you use this repository in your work, cite the corresponding paper or project source that this implementation is based on, and describe your drug-food adaptation clearly.
