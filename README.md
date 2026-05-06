<div align="center">

```
 ██████╗██╗  ██╗ █████╗ ██████╗      ████████╗██████╗  █████╗ ███╗   ██╗███████╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗██████╗ 
██╔════╝██║  ██║██╔══██╗██╔══██╗        ╚══██╔╝██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔══██╗
██║     ███████║███████║██████╔╝           ██║ ██████╔╝███████║██╔██╗ ██║███████╗█████╗  ██║   ██║██████╔╝██╔████╔██║█████╗  ██████╔╝
██║     ██╔══██║██╔══██║██╔══██╗           ██║ ██╔══██╗██╔══██║██║╚██╗██║╚════██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ██╔══██╗
╚██████╗██║  ██║██║  ██║██║  ██║           ██║ ██║  ██║██║  ██║██║ ╚████║███████║██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗██║  ██║
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝           ╚═╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
```

# 🔤 Character-Level Transformer

### *A sequence-to-sequence Transformer that thinks one character at a time.*

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)]()

</div>

---

## 📖 What Is This?

Most language models operate at the **word** or **subword** level — they see tokens like `"hello"` or `"transformer"`. This project takes a different, more fundamental approach: it processes text **one character at a time**.

The **Character-Level Transformer** is a from-scratch implementation of the classic encoder-decoder Transformer architecture in PyTorch, applied to character-level sequence-to-sequence tasks. Every single letter, space, and punctuation mark is its own token. No tokenizer needed.

This makes it a perfect learning ground for understanding:
- How Transformers truly work under the hood
- The interplay between encoder and decoder
- What raw language looks like to a neural network

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧱 **Scratch Implementation** | Full Transformer built in pure PyTorch — no HuggingFace, no shortcuts |
| 🔤 **Character Tokenization** | Each character is its own token. Simple, transparent, zero preprocessing |
| 🔁 **Seq2Seq Architecture** | Encoder–Decoder design for tasks like transliteration, transcription, or text transformation |
| 🏋️ **Custom Training Loop** | AdamW optimizer with cross-entropy loss and teacher forcing |
| 🔮 **Inference Module** | Dedicated script for generating predictions from a trained model |
| 📦 **Clean Codebase** | Neatly separated into `model`, `dataset`, `train`, and `inference` modules |

---

## 🗂️ Project Structure

```
character-level-transformer/
│
├── model.py          # 🧠  The Transformer model — Multi-Head Attention, 
│                     #     Positional Encoding, Encoder & Decoder stacks
│
├── dataset.py        # 📂  Dataset class — character-to-index mapping,
│                     #     padding, and DataLoader preparation
│
├── train.py          # 🏋️  Training loop — AdamW optimizer, CrossEntropyLoss,
│                     #     teacher forcing, epoch/batch logging
│
├── inference.py      # 🔮  Inference — load a checkpoint and generate 
│                     #     character-by-character predictions
│
└── .gitignore        # 🚫  Standard Python ignores
```

---

## 🧠 Architecture Deep Dive

The model follows the original **"Attention Is All You Need"** architecture (Vaswani et al., 2017), adapted for character-level input.

```
INPUT SEQUENCE (characters)
        │
        ▼
┌───────────────────┐
│  Character Embed  │  ← Each char → dense vector
│  + Pos. Encoding  │  ← Inject sequence position info
└───────────────────┘
        │
        ▼
┌───────────────────┐
│    ENCODER        │
│  ┌─────────────┐  │
│  │ Multi-Head  │  │  ← Attends to all source chars
│  │  Attention  │  │
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │  Feed Fwd   │  │  ← Position-wise transformation
│  └─────────────┘  │
│    × N layers     │
└───────────────────┘
        │ (memory)
        ▼
┌───────────────────┐
│    DECODER        │
│  ┌─────────────┐  │
│  │ Masked Attn │  │  ← Can't peek at future chars
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │ Cross-Attn  │  │  ← Attends to encoder memory
│  └─────────────┘  │
│  ┌─────────────┐  │
│  │  Feed Fwd   │  │
│  └─────────────┘  │
│    × N layers     │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Linear + Softmax │  ← Project to target vocab
└───────────────────┘
        │
        ▼
OUTPUT SEQUENCE (characters)
```

### Why Character-Level?

| Aspect | Word-Level | Character-Level |
|---|---|---|
| Vocabulary size | 30,000–50,000+ | ~100–200 |
| Handles rare words | ❌ OOV issues | ✅ Always composable |
| Sequence length | Shorter | Longer |
| Learning difficulty | Easier | More expressive |
| Interpretability | Lower | Higher |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/youssofhossam/character-level-transformer.git
cd character-level-transformer

# 2. Install dependencies
pip install torch
```

### Training

```python
from model import Transformer
from dataset import CharDataset
from train import train_model
from torch.utils.data import DataLoader

# Load your data
dataset = CharDataset(your_source_texts, your_target_texts)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = Transformer(
    src_vocab_size=...,
    trgt_vocab_size=...,
    d_model=256,
    nhead=8,
    num_layers=4
)

# Train
train_model(model, dataloader, lr=1e-4)
```

### Inference

```python
from inference import generate

output = generate(model, input_string="hello world", max_len=100)
print(output)
```

---

## 🏋️ Training Details

The training loop (`train.py`) follows best practices for sequence-to-sequence learning:

- **Optimizer**: AdamW — weight decay regularization for better generalization
- **Loss**: Cross-Entropy with `ignore_index=0` to properly handle padding tokens
- **Teacher Forcing**: During training, the decoder receives the ground-truth previous character as input, stabilizing early learning
- **Epochs**: 20 default training epochs
- **Logging**: Loss printed every 50 batches for live monitoring

```
Starting Training...
Epoch 1  | Batch 0/500   | Loss: 4.2301
Epoch 1  | Batch 50/500  | Loss: 3.1842
Epoch 1  | Batch 100/500 | Loss: 2.7214
...
```

---

## 💡 Potential Use Cases

This architecture can be applied to a wide range of character-level sequence transformation tasks:

- **Transliteration** — Convert text between writing systems (e.g., Arabic ↔ Latin)
- **Spelling Correction** — Map misspelled → correctly spelled words
- **Text Normalization** — Standardize abbreviations, casing, punctuation
- **Cipher Decoding** — Learn character substitution mappings
- **Language Transcription** — Convert phonetic to standard orthography
- **Name Generation** — Train on name datasets to generate new names

---

## 📁 Data Format

The model expects parallel character sequences — a source string and a target string for each training example.

```
Source: "h e l l o"   →   Target: "H E L L O"
Source: "merhba"      →   Target: "welcome"
Source: "helo wrold"  →   Target: "hello world"
```

The `dataset.py` module handles:
- Building character vocabularies from your data
- Mapping characters to integer indices
- Adding `<PAD>`, `<SOS>`, and `<EOS>` special tokens
- Creating batched tensors for the DataLoader

---

## 🔬 How the Training Loop Works

```python
for epoch in range(20):
    for src, trgt in dataloader:
        
        src_input    = src              # Full source sequence
        trgt_input   = trgt[:, :-1]     # Target without last char (teacher forcing input)
        trgt_expected = trgt[:, 1:]     # Target without first char (what we predict)
        
        output = model(src_input, trgt_input)
        
        loss = CrossEntropyLoss(
            output.view(-1, vocab_size),
            trgt_expected.view(-1)
        )
        
        loss.backward()
        optimizer.step()
```

The key insight: the decoder sees `trgt[:, :-1]` (e.g., `<SOS> h e l l`) and must predict `trgt[:, 1:]` (e.g., `h e l l o`). This is **teacher forcing** — using ground truth inputs during training for stable gradients.

---

## 🤝 Contributing

Contributions, ideas, and improvements are warmly welcome! Here's how:

1. **Fork** the repository
2. **Create** your feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

Some ideas for contributions:
- [ ] Add beam search decoding to `inference.py`
- [ ] Implement learning rate scheduling
- [ ] Add evaluation metrics (character error rate, accuracy)
- [ ] Support for loading pre-trained checkpoints
- [ ] Example notebook with a real dataset

---

## 📚 References & Further Reading

- [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017) — The original Transformer paper
- [**The Annotated Transformer**](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — Harvard NLP — Line-by-line walkthrough
- [**Character-Level Language Models**](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) — Andrej Karpathy — The classic intuition builder

---

<div align="center">

**Built with ❤️ and PyTorch**

*If this project helped you learn something new, consider giving it a ⭐*

</div>
