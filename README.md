# Genetic Programming-Based Language Model Evolution
### Autonomous NLP Architecture Optimization — Next Word Prediction

---

## Overview

This project uses **Genetic Programming (GP)** to automatically evolve neural network architectures for **Next Word Prediction** — without any human manually designing the model.

GP generates, tests, and breeds architectures over multiple generations, just like biological natural selection, until it finds the best performing one.

---

## Dataset — WikiText-2

- **Source:** Verified English Wikipedia articles
- **Task:** Predict the next word in a sentence
- **Training words:** ~2 million
- **Vocabulary:** 44,563 unique words
- **Loaded via:** HuggingFace `datasets` library

```python
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
```

---

## How GP Was Applied

Each architecture is represented as a **tree** built from these building blocks:

| Block             | Options                                       |
|---                |---                                            |
| Recurrent layers  | LSTM, GRU (64 / 128 / 256 units)              |
| Attention layers  | Transformer, Attention (64 / 128 / 256 units) |
| Feed Forward      | FFN (64 / 128 / 256 units)                    |
| Regularization    | Dropout (0.1 / 0.2 / 0.3 / 0.5)               |
| Normalization     | LayerNorm                                     |
| Activations       | ReLU, GELU, Tanh                              |

**GP Loop:**
```
1. Generate 20 random architectures
2. Train each for 2 epochs → measure Perplexity
3. Keep best 5 (lower perplexity = better)
4. Breed them via Crossover + Mutation
5. Repeat for 5 generations
```

---

## GP Evolution Results

| Generation | Best Perplexity | Avg Perplexity |
|---         |---              |---             |
| 1          | 2.73            | 607.92         |
| 2          | 2.57            | 386.56         |
| 3          | 2.57            | 219.97         |
| 4          | 1.79            | 148.41         |
| 5          | **1.75**        | 131.95         |

GP consistently improved across generations — average perplexity dropped from **607 → 131**.

---

## Best Architecture Found

```
GRU256(DROP03(LN(TRANS256(INPUT))))
```

**Flow:**
```
Input → Transformer(256, 8 heads) → LayerNorm → Dropout(30%) → GRU(256) → Next Word
```

GP independently discovered that **Transformer + GRU** outperforms all other combinations.

**Final model after full training:**
```
Validation Perplexity : 1.1114
Parameters            : 18,768,787
```

> Perplexity of 1.1114 means the model narrows down 44,563 possible next words to essentially 1 correct prediction.

---

## Output — Live Prediction Demo

The model predicts the **top 3 most likely next words** for any input sentence.

```
Input  : "The doctors concluded that"
Output :
  1. the      16.76%  ████████
  2. "          3.82%  █
  3. it         3.48%  █
```

---

## How to Run

**Requirements:**
```bash
!pip install datasets transformers deap -q
```

**Steps:**
1. Open `GP_NLP_Colab.ipynb` in Google Colab
2. Set runtime to GPU → `Runtime → Change Runtime Type → GPU`
3. Run all cells from **Cell 1 to Cell 14** in order
4. Cell 14 launches the live prediction demo

**If Colab disconnects:**
- Rerun Cells 1–11
- Cell 11 auto-resumes from the latest saved checkpoint

---

## How to Give Input

Run **Cell 14** and type any sentence when prompted:

```
Your input: The stock market in London was
```

**Tips:**
- Use at least **5 words** for meaningful predictions
- Use **proper capitalization** (London not london)
- For best results provide **64 words** (full context window)
- Type `check: your text` to verify words are in vocabulary
- Type `quit` to exit

---

## Author

**Sai Abhi Ram** — [@SaiAbhiRam9496](https://github.com/SaiAbhiRam9496)