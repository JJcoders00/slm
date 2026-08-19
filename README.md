# JJ Coders SLM (Small Language Model)

A parameter-efficient, context-anchored Small Language Model architecture engineered from scratch in PyTorch. 

Built on the principle of **computational neural architecture**—maximizing reasoning, domain specialization, and generative capacity per parameter on constrained consumer hardware using **Recurrent Computational Depth**, **Latent Context Anchoring**, **Rotary Position Embeddings (RoPE)**, **RMSNorm**, and **SwiGLU** feed-forward networks.

---

## 🔬 Architectural Specifications

| Parameter | Specification |
| :--- | :--- |
| **Physical Parameter Footprint** | **~55 Million to ~100 Million** |
| **Effective Computational Depth** | **16 to 20 Layers** (via Weight-Tied Recurrent Loops) |
| **Hidden Dimension ($d_\text{model}$)** | 512 – 768 |
| **Attention Heads** | 8 – 12 (Head Dimension: 64) |
| **Vocabulary Size** | 8,192 (Custom Byte-Pair Encoding) |
| **Context Window** | 512 Tokens |
| **Normalization** | Root Mean Square Normalization (RMSNorm) |
| **Position Embeddings** | Rotary Position Embeddings (RoPE) |
| **Activation Function** | SwiGLU ($2.67 \times d_\text{model}$) |
| **Training Precision** | Mixed Precision (FP16 with Dynamic Gradient Scaling) |
| **Memory Optimization** | Zero-RAM Binary Memory-Mapping (`np.memmap`) |
| **Active VRAM Footprint** | ~8.5 GB – 11.5 GB (NVIDIA T4 Compatible) |

---

## 💡 Core Engineering Innovations

### 1. Latent Context Anchoring (Zero-Drift Subject Injection)
Small language models often suffer from attention dilution during longer generations. JJ Coders SLM computes a global latent representation of the input prompt and continuously injects it into the residual stream of all attention blocks, maintaining topic focus and preventing semantic drift across multi-paragraph responses.

### 2. Recurrent Computational Depth
Rather than increasing physical parameter bloat, the architecture routes intermediate latent representations through physical transformer blocks across multiple recurrent passes. This achieves the representational capacity of a **16-to-20 layer foundation model** while keeping the physical memory footprint compact (~55M–100M parameters).

### 3. Masked Supervised Fine-Tuning (SFT)
Prompt tokens are assigned a target index of `-100` during optimization (`CrossEntropyLoss(ignore_index=-100)`). Gradients are evaluated **100% on the accuracy of the generated responses**, eliminating prompt echoing and topic confusion.

### 4. Precision Anti-Repetition Nucleus Inference
Combines dynamic repetition penalty ($1.20 - 1.25$), temperature calibration ($0.25 - 0.35$), and Top-p ($0.9$) nucleus sampling to suppress degenerative word loops and produce deterministic, syntactically valid code and explanations.

---

## 📊 Evaluation & Domain Benchmark Highlights

### 1. Planetary Science & Astronomy
```text
User: The solar system consists of the Sun and eight planets. Describe the inner and outer planets.
JJ AI: Our Solar System is divided into two main planetary regions:

1. Terrestrial (Inner) Planets:
- Mercury: The smallest planet and closest to the Sun; has extreme temperature fluctuations and no significant atmosphere.
- Venus: Similar in size to Earth; has a dense carbon dioxide atmosphere producing extreme greenhouse heat.
- Earth: The only known planet with abundant liquid surface water and life.
- Mars: Known as the Red Planet due to iron oxide; features Olympus Mons and polar ice caps.

2. Giant (Outer) Planets:
- Jupiter: The largest planet, famous for its Great Red Spot storm and dozens of moons.
- Saturn: Renowned for its prominent, extensive ring system of ice and rock.
- Uranus: An ice giant with a unique sideways rotation axis.
- Neptune: The most distant major planet, known for supersonic methane winds.
