# LoRA: Low-Rank Adaptation of Large Language Models (From Scratch in PyTorch)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

An implementation from first principles of **Low-Rank Adaptation (LoRA)** for Parameter-Efficient Fine-Tuning (PEFT) of Transformers and Large Language Models in PyTorch, following [Hu et al., 2021 (*LoRA: Low-Rank Adaptation of Large Language Models*)](https://arxiv.org/abs/2106.09685).

---

> 🇫🇷 **Note aux recruteurs francophones** : Un résumé rapide du projet en français est disponible en fin de document — [Consulter le résumé en français](#-résumé-rapide-en-français).

---

## 📌 Executive Summary & Key Results

Instead of fine-tuning all 108M parameters of BERT for downstream sentiment classification (IMDB), this project implements low-rank weight decomposition matrices injected directly into multi-head self-attention projections ($W_q, W_k, W_v$).

### 📊 Benchmark Comparison (BERT-base-cased on IMDB)

| Adaptation Method | Total Parameters | Trainable Parameters | % Trainable | Test Accuracy | Train Runtime | Inference Latency Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Fine-Tuning** | 108,311,810 | 108,311,810 | 100.0% | **90.9%** | 554 s | None |
| **Our LoRA from Scratch ($r=8$)** | 108,755,716 | **443,906** | **0.41%** | **87.9%** | **464 s** | **0 ms** (via weight merge) |
| **Hugging Face PEFT ($r=1$)** | 108,350,212 | 38,402 | 0.035% | 69.9% | 468 s | None |

> **Key Takeaway**: With **less than 0.5%** of parameters trained, our custom LoRA ($r=8$) achieves **87.9% accuracy** (within 3% of full model fine-tuning) while reducing optimizer memory footprint by orders of magnitude and eliminating storage overhead for downstream task checkpoints.

---

## 📐 Mathematical Formulation

### 1. The Low-Rank Hypothesis
Over-parameterized foundation models reside on a low intrinsic dimension manifold during task-specific adaptation ([Aghajanyan et al., 2020](https://arxiv.org/abs/2012.13255)). Consequently, weight update matrices $\Delta W$ have a low intrinsic rank $r \ll \min(d_{in}, d_{out})$.

### 2. Decomposition
For a pretrained linear projection layer with weights $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$, the adapted forward pass computes:

$$\mathbf{h} = W_0 \mathbf{x} + \Delta W \mathbf{x} = W_0 \mathbf{x} + \frac{\alpha}{r} (B A) \mathbf{x}$$

where:
- $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$ is **frozen** (requires_grad = False).
- $A \in \mathbb{R}^{r \times d_{in}}$ is the down-projection matrix initialized via Kaiming uniform distribution.
- $B \in \mathbb{R}^{d_{out} \times r}$ is the up-projection matrix **initialized to zeros**.
- $r$ is the adaptation rank ($r \ll d$).
- $\alpha$ is a scaling hyperparameter (constant relative to $r$), stabilizing optimization when varying rank $r$.

```
               x  (Input Vector: in_dim)
               │
       ┌───────┴───────┐
       ▼               ▼
   [ W_0 ]          [   A   ]   (r x in_dim: Kaiming Init)
 (Frozen Base)         │
       │               ▼
       │            [   B   ]   (out_dim x r: Zero Init)
       │               │
       │               ▼
       │         * (alpha / r)
       │               │
       └───────┬───────┘
               ▼
             ( + )
               │
               ▼
               h  (Output Vector: out_dim)
```

### 3. Critical Initialization Invariant
Because $B = 0$ at initialization:

$$\Delta W_{t=0} = \frac{\alpha}{r} (0 \cdot A) = 0$$

At step $0$, the adapted model outputs **identically** to the base pretrained model, guaranteeing seamless transition into fine-tuning without initial distortion.

### 4. Zero-Latency Inference Merging
At deployment time, the learned adapter weights can be permanently folded into the frozen base weights:

$$W_{\text{deploy}} = W_0 + \frac{\alpha}{r} B A$$

This eliminates any auxiliary memory lookups or latency penalties during production serving.

---

## 🛠️ Repository Structure

```
├── lora.py                 # Clean PyTorch implementation of LoRALinear & model adapters
├── lora.ipynb              # Complete end-to-end training notebook with full evaluation
├── README.md               # Mathematical formulation, architecture & benchmark results
└── .gitignore
```

---

## 💻 Quickstart

### 1. Installation
```bash
git clone https://github.com/nourou-dine-cisse/LoRA-From-Scratch-PyTorch.git
cd LoRA-From-Scratch-PyTorch
pip install torch transformers datasets evaluate scikit-learn
```

### 2. Python Usage Example
```python
import torch
import torch.nn as nn
from lora import LoRALinear, linear_to_lora

# Replace standard linear layer with LoRA
layer = nn.Linear(768, 768)
lora_layer = linear_to_lora(layer, rank=8, lora_alpha=16.0)

# Input tensor
x = torch.randn(4, 32, 768)
output = lora_layer(x)

# Fold weights for zero-latency production inference
lora_layer.merge()
inference_output = lora_layer(x)
```

### 3. Running the Jupyter Notebook
Open `lora.ipynb` in Jupyter Lab or VS Code to reproduce the comparative benchmark between full fine-tuning, custom LoRA, and Hugging Face PEFT.

---

## 🔬 Implementation Highlights

1. **Target Attention Projections**: Adapted Query ($W_q$), Key ($W_k$), and Value ($W_v$) projections across all 12 BERT self-attention layers:
   $$\text{Adapted Layers} = 12 \times 3 \times 8 \times (768 + 768) = 442,368 \text{ parameters}$$
   Plus the classification head (1,538 parameters), totaling **443,906 trainable parameters**.
2. **Selective Gradient Propagation**: Rigorous inspection ensuring only `lora_a`, `lora_b`, and task classifier retain `requires_grad = True`.
3. **Reproducibility**: Uniform training arguments (2 epochs, 5,000 samples, fixed random seed 42) across all methods for strictly fair comparison.

---

## 📚 References
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv preprint [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. arXiv preprint [arXiv:2012.13255](https://arxiv.org/abs/2012.13255).
- Vaswani, A., et al. (2017). *Attention Is All You Need*. NeurIPS 2017.

---
---

## 🇫🇷 Résumé Rapide en Français

Ce projet propose une implémentation complète from scratch sous **PyTorch** de la technique **LoRA (Low-Rank Adaptation)** pour le fine-tuning économe en paramètres (**PEFT**) de modèles Transformers et Grands Modèles de Langage (LLMs), suivant les principes de [Hu et al., 2021](https://arxiv.org/abs/2106.09685).

Plutôt que d'adapter l'intégralité des 108 millions de paramètres de BERT pour la classification de sentiment sur IMDB :
- Nous gelons 99,6 % des poids du modèle de base.
- Nous injectons une décomposition matricielle de bas rang $W = W_0 + \frac{\alpha}{r} BA$ sur les projections d'attention ($W_q, W_k, W_v$).
- À l'inférence, les matrices entraînées sont fusionnées directement dans les poids de base, garantissant **une surcharge de latence strictement nulle (0 ms)**.

**Résultat clé** : avec seulement **0,41 % de paramètres entraînés** (443 906 paramètres), notre modèle atteint **87,9 % de précision** sur le jeu de test IMDB, rivalisant avec les 90,9 % du fine-tuning intégral tout en divisant drastiquement l'empreinte mémoire d'entraînement.

---

## 👤 Author
**Mouhamed Nourou Dine Cisse**  
- Diplôme d'Ingénieur in Applied Mathematics & Computer Science, **ENSIIE**
- Master 2 in Artificial Intelligence, **Université Paris-Saclay** (LISN CNRS / DATAIA)
- Portfolio: [nourouportfoliowebsite.vercel.app](https://nourouportfoliowebsite.vercel.app)
- GitHub: [@nourou-dine-cisse](https://github.com/nourou-dine-cisse)
- LinkedIn: [mouhamed-nourou-dine-cisse](https://linkedin.com/in/mouhamed-nourou-dine-cisse-094159327)
