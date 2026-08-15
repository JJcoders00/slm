# JJ Coders SLM

A parameter-efficient Small Language Model (SLM) engineered from scratch in PyTorch, utilizing **Latent Context Anchoring**, **Recurrent Computational Depth**, **Rotary Position Embeddings (RoPE)**, **RMSNorm**, and **SwiGLU** feed-forward networks.

---

## Architectural Specifications

| Parameter | Specification |
| :--- | :--- |
| **Physical Parameter Count** | ~22 Million |
| **Effective Computational Depth** | 12 Layers (4 physical layers × 3 recurrent loops) |
| **Vocabulary Size** | 8,192 (Byte-Pair Encoding) |
| **Hidden Dimension ($d_\text{model}$)** | 384 |
| **Attention Heads** | 6 (Head Dimension: 64) |
| **Context Window** | 512 Tokens |
| **Normalization** | Root Mean Square Normalization (RMSNorm) |
| **Position Embeddings** | Rotary Position Embeddings (RoPE) |
| **Activation Function** | SwiGLU |

---

## Key Design Principles

1. **Latent Context Anchoring:** A global latent vector derived from the prompt is injected into the residual stream across attention blocks to maintain semantic focus and mitigate topic drift during generation.
2. **Recurrent Depth:** Loops latent states across shared physical blocks, increasing representational depth without increasing the parameter footprint.
3. **Masked Supervised Fine-Tuning (SFT):** Targets prompt tokens with an index of `-100` during optimization, evaluating loss strictly on response tokens.

---

## Quickstart

### 1. Installation
```bash
git clone [https://github.com/jjcoders00/slm.git](https://github.com/jjcoders00/slm.git)
cd slm
pip install -r requirements.txt
