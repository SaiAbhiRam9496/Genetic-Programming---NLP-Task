# Genetic Programming-Based Language Model Evolution
### Autonomous NLP Architecture Optimization — Next Word Prediction

---

## Overview

This project uses **Genetic Programming (GP)** to automatically evolve neural network architectures for **Next Word Prediction** — without any human manually designing the model. GP generates, tests, and breeds architectures over multiple generations — inspired by biological natural selection — until it discovers the best performing architecture automatically.

---

## Final Results

| Metric | Value |
|---|---|
| Task | Next Word Prediction |
| Dataset | WikiText-2 |
| GP Generations | 5 |
| Population per Generation | 20 architectures |
| Best GP Phase Perplexity | 1.75 (Generation 5) |
| **Final Model Perplexity** | **1.1056** |
| Best Architecture | `GRU256(DROP03(LN(TRANS256)))` |
| Total Parameters | 18,768,787 |
| Training Stopped | Epoch 5 (Early Stopping) |

> **Perplexity of 1.1056** means the model narrows down 44,563 possible next words to essentially 1 correct prediction — an outstanding result achieved through GP-guided architecture search.

---

## Dataset — WikiText-2

- **Source:** Verified English Wikipedia articles
- **Topics:** History, Science, Sports, Geography, Technology, Biography
- **Training words:** ~2 million
- **Vocabulary:** 44,563 unique words
- **Loaded via:** HuggingFace `datasets` library

```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```

| Split | Samples | Words |
|---|---|---|
| Train | 23,767 | 2,051,910 |
| Validation | 2,461 | 213,886 |
| Test | 2,891 | 241,211 |

---

## How GP Was Applied

Each architecture is represented as a **tree** built from these building blocks:

| Block | Options |
|---|---|
| Recurrent | LSTM, GRU (64 / 128 / 256 units) |
| Attention | Transformer, Attention (64 / 128 / 256 units) |
| Feed Forward | FFN (64 / 128 / 256 units) |
| Regularization | Dropout (0.1 / 0.2 / 0.3 / 0.5) |
| Normalization | LayerNorm |
| Activations | ReLU, GELU, Tanh |

**GP Loop:**
```
1. Generate 20 random architecture trees
2. Train each for 2 epochs → measure Perplexity
3. Select best via Tournament Selection
4. Keep top 2 unchanged (Elitism)
5. Breed remaining via Crossover + Mutation
6. Repeat for 5 generations
7. Best architecture → Full training with Early Stopping
```

---

## GP Evolution Results

| Generation | Best Perplexity | Avg Perplexity | Best Architecture |
|---|---|---|---|
| 1 | 2.73 | 607.92 | `GRU256(TRANS128)` |
| 2 | 2.57 | 386.56 | `GRU256(TRANS128(FFN128))` |
| 3 | 2.57 | 219.97 | `GRU256(TRANS128(FFN128))` |
| 4 | 1.79 | 148.41 | `GRU256(TRANS128(DROP03(LN(TRANS256))))` |
| 5 | **1.75** | 131.95 | `GRU256(DROP03(LN(TRANS256)))` |

Average population perplexity dropped from **607.92 → 131.95** across generations — confirming GP evolution was directional and effective.

---

## Full Training Results

| Epoch | Train Loss | Val Loss | Val Perplexity |
|---|---|---|---|
| 1 | 0.1773 | 0.1056 | 1.1114 |
| 2 | 0.1089 | 0.1003 | **1.1056** ← Best |
| 3 | 0.1080 | 0.1012 | 1.1065 |
| 4 | 0.1076 | 0.1013 | 1.1066 |
| 5 | 0.1078 | 0.1019 | 1.1072 |

Early stopping triggered at Epoch 5. Best model saved at **Epoch 2**.

---

## Best Architecture Found

```
GRU256(DROP03(LN(TRANS256(INPUT))))
```

**Execution flow:**
```
Input Words
    ↓
Embedding Layer (128 dimensions)         ← Fixed
    ↓
Transformer Encoder (256 units, 8 heads) ← GP Evolved
    ↓
Layer Normalization                      ← GP Evolved
    ↓
Dropout (30%)                            ← GP Evolved
    ↓
GRU (256 units)                          ← GP Evolved
    ↓
Output Layer (44,563 vocabulary)         ← Fixed
    ↓
Top 3 Next Word Predictions
```

GP independently discovered that **Transformer → LayerNorm → Dropout → GRU** outperforms all other combinations tested across 5 generations and 100 total architectures.

---

## How to Run

### Requirements
```bash
!pip install datasets transformers deap -q
```

### Steps
1. Open `GP_NLP_Colab.ipynb` in **Google Colab**
2. Set runtime → `Runtime → Change Runtime Type → GPU`
3. Run cells **in order** from Cell 1 to Cell 14

### Cell Guide

| Cell | Description | Est. Time |
|---|---|---|
| Cell 1 | GPU Check + Google Drive Mount | Instant |
| Cell 2 | Install Libraries | 1 min |
| Cell 3 | Load WikiText-2 Dataset | 1 min |
| Cell 4 | Explore Dataset | Instant |
| Cell 5 | Tokenization + Vocabulary | 1 min |
| Cell 6 | DataLoader Setup | Instant |
| Cell 7 | DEAP GP Setup + Primitive Set | Instant |
| Cell 8 | Tree → PyTorch Model Converter | Instant |
| Cell 9 | Fitness Function (Proxy Eval) | Instant |
| Cell 10 | Save / Load Utilities | Instant |
| Cell 11 | GP Main Loop (5 Generations) | ~2 hours |
| Cell 12 | Full Training of Best Architecture | ~2 hours |
| Cell 13a | Load Best Model | Instant |
| Cell 13b | GP Evolution Graphs | Instant |
| Cell 13c | Training Loss Graphs | Instant |
| Cell 14 | Live Top 3 Prediction Demo | Interactive |

### If Colab Disconnects
- Rerun Cells 1 to 11 only
- Cell 11 **automatically resumes** from latest saved checkpoint
- No progress lost — all checkpoints saved to Google Drive after every generation

---

## Project Structure

```
GP_NLP_Project/  (Google Drive)
├── checkpoints/
│   ├── checkpoint_gen1.pkl
│   ├── checkpoint_gen2.pkl
│   ├── checkpoint_gen3.pkl
│   ├── checkpoint_gen4.pkl
│   └── checkpoint_gen5.pkl
├── models/
│   ├── best_model_gen1.pt → gen5.pt
│   └── final_best_model.pt  ← Perplexity 1.1056
└── logs/
    ├── gp_evolution_log.json
    ├── final_training_log.json
    ├── graph_gp_evolution.png
    ├── graph_training_loss.png
    ├── graph_val_perplexity.png
    └── graph_best_per_generation.png
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| PyTorch | Neural network building and training |
| DEAP | Genetic Programming framework |
| HuggingFace Datasets | WikiText-2 loading |
| Matplotlib | Graph generation |
| Google Colab (Tesla T4 — 15.64GB) | GPU runtime |
| Google Drive | Checkpoint and model storage |

## Author

**Sai Abhi Ram** — [@SaiAbhiRam9496](https://github.com/SaiAbhiRam9496)

---

*Genetic Programming × Neural Architecture Search × Next Word Prediction*