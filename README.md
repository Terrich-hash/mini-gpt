 # **MiniGPT – Pure Python GPT (No Dependencies)**

The most atomic way to train and run a GPT model in pure Python.
One file. No frameworks. No NumPy. No PyTorch. Just math.

---

This project implements a minimal GPT-style transformer from scratch, including:
- Character-level tokenizer
- Custom autograd engine
- Multi-head self-attention
- RMSNorm
- MLP block
- Adam optimizer
- Training loop
- Inference sampling

Everything is written in pure Python using only the standard library.

# Features

- Character-level language modeling
- Custom scalar-based autograd (Value class)
- Transformer block (Attention + MLP)
- RMSNorm instead of LayerNorm
- ReLU instead of GELU
- Adam optimizer with learning rate decay
- Temperature-based sampling

# Architecture Overview
Model Hyperparameters
```
n_embd = 16      # embedding dimension
n_head = 4       # number of attention heads
n_layer = 1      # transformer layers
block_size = 16  # max context length
```

# Transformer Block Structure
```
Input
  │
  ├── Token Embedding
  ├── Position Embedding
  │
  ├── RMSNorm
  │
  ├── Multi-Head Self Attention
  │       ├── Q projection
  │       ├── K projection
  │       ├── V projection
  │       ├── Scaled dot-product attention
  │       └── Output projection
  │
  ├── Residual Connection
  │
  ├── RMSNorm
  │
  ├── MLP
  │       ├── Linear
  │       ├── ReLU
  │       └── Linear
  │
  ├── Residual Connection
  │
  └── Final Linear (LM Head)
```

# Autograd Engine

The Value class implements:
- Forward scalar computation
- Automatic graph construction
- Reverse-mode backpropagation

Supported operations:

`+ `
`-`
`*`
`/`
`**`
`exp()`
`log()`
`relu()`

Backprop works via topological sorting of the computation graph.

# Dataset
The script automatically downloads a dataset of names:
```
https://raw.githubusercontent.com/karpathy/makemore/master/names.txt
```
Each name is treated as a training document.

Training objective:
- Predict the next character given previous characters.

# How To Run
1️⃣ Clone
```
git clone https://github.com/Terrich-hash/mini-gpt.git
cd mini-gpt
```
2️⃣ Run
```
python3 micro_gpt.py
```

That’s it.
- No installs.
- No virtual environments.
- No dependencies.

# Learning Path
If you understand this file, you understand:
- Autograd
- Attention
- Transformer blocks
- Language modeling
- Optimizers
- Sampling

That’s 80% of modern LLM architecture.

