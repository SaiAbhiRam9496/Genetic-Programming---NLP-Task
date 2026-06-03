# Genetic Programming for Neural Architecture Search on WikiText-2

## Project Overview
This repository implements **Genetic Programming (GP)** combined with **Neural Architecture Search (NAS)** to evolve and optimize deep learning architectures for language modeling on the **WikiText-2** dataset. The project compares baseline models against GP-evolved architectures.

## Key Components

### 1. **BaseLine.ipynb**
Establishes baseline performance using hand-crafted architectures:
- **PlainGRU** — Gated Recurrent Unit (2 layers, 128 hidden dim)
- **PlainLSTM** — Long Short-Term Memory (2 layers, 128 hidden dim)  
- **PlainTransformer** — Transformer encoder stack (2 layers, 4 attention heads)

Training: 10 epochs on WikiText-2 with early stopping, sequence length 64, batch size 64.

### 2. **GP_NAS_WikiText2.ipynb**
Uses genetic programming to evolve optimal architectures:
- **Population Size**: 20 individuals | **Generations**: 10
- **Building Blocks**: Fusion layers, TransformerBlocks, AttentionBlocks, GRU/LSTM cells
- **Crossover Rate**: 50% | **Mutation Rate**: 30%
- **Constraints**: Max depth 4, param budget 8M, proxy training on 20% data
- **Proxy Fitness**: 2-3 epochs early, 10 epochs final validation

## Dataset & Configuration
- **Dataset**: WikiText-2 (raw version)
- **Vocabulary**: Tokens with frequency ≥ 3, size ~33K
- **Sequence Length**: 64 tokens
- **Batch Size**: 64
- **Word Embedding**: 100-dim | **POS Embedding**: 16-dim
- **Hidden Size**: 128 | **Dropout**: 0.3

## Architecture Search Details
- **Crossover**: Single-point tree-based genetic crossover
- **Mutation**: Subtree replacement from initialized trees
- **Elitism**: Top 2 individuals preserved each generation
- **Tournament Selection**: Size 3 for parent selection
- **Size Penalty**: 0.5 weight to balance accuracy vs. model efficiency

## Dependencies
```
torch, torchvision, torchtext
transformers, datasets
numpy, pandas, scikit-learn
spacy (en_core_web_sm)
```

## Environment
- **Designed for**: Google Colab (GPU required)
- **Device**: CUDA GPU / CPU fallback
- **Storage**: Google Drive mounted for checkpoints & results

## Results
- Baseline models establish performance benchmarks
- GP-evolved architectures aim for accuracy-efficiency Pareto frontier
- All results saved to JSON logs and model checkpoints

## File Structure
```
├── BaseLine.ipynb           (Baseline model training)
├── GP_NAS_WikiText2.ipynb   (Genetic Program architecture search)
├── Updated_GP_Project.pdf   (Technical documentation)
└── README.md                (This file)
```

## Usage
1. Open notebooks in Google Colab
2. Mount Google Drive when prompted
3. Run cells sequentially (dependencies installed automatically)
4. Monitor training via console logs and saved checkpoints
5. Compare baseline vs. evolved architectures in results JSON

## Authors & Citation
Genetic Programming Neural Architecture Search for Language Modeling on WikiText-2.
For details see `Updated_GP_Project.pdf`.
