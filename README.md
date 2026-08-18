# JJ Coders SLM (Small Language Model)

A parameter-efficient, context-anchored Small Language Model engineered from scratch in PyTorch. Designed on the principle of **computational neural architecture**—maximizing reasoning capacity per parameter on constrained hardware using **Recurrent Computational Depth**, **Latent Context Anchoring**, **Rotary Position Embeddings (RoPE)**, **RMSNorm**, and **SwiGLU** feed-forward networks.

---

## 🔬 Architectural Specifications

| Parameter | Specification |
| :--- | :--- |
| **Physical Parameters** | **~100.4 Million** |
| **Effective Computational Depth** | **20 Layers** (10 physical layers × 2 recurrent loops) |
| **Hidden Dimension ($d_\text{model}$)** | 768 |
| **Attention Heads** | 12 (Head Dimension: 64) |
| **Vocabulary Size** | 8,192 (Custom Byte-Pair Encoding) |
| **Context Window** | 512 Tokens |
| **Normalization** | Root Mean Square Normalization (RMSNorm) |
| **Position Embeddings** | Rotary Position Embeddings (RoPE) |
| **Activation Function** | SwiGLU ($2.67 \times d_\text{model}$) |
| **Training Precision** | Mixed Precision (FP16 with Dynamic Gradient Scaling) |
| **Active VRAM Footprint** | ~9.5 GB – 10.5 GB (T4 GPU Compatible) |

---

## 💡 Key Architectural Innovations

### 1. Latent Context Anchoring (Zero-Drift Subject Injection)
Standard small transformers often suffer from attention dilution over multi-paragraph outputs. JJ Coders SLM computes a global latent representation of the input prompt and continuously injects it into the residual stream of all attention blocks, maintaining topic focus and preventing semantic drift.

### 2. Recurrent Computational Depth
Rather than scaling parameter bloat, the architecture routes intermediate latent representations through physical transformer blocks across multiple recurrent passes. This provides the representational power of a **20-layer foundation model** while keeping the physical memory footprint at **~100M parameters**.

### 3. Masked Supervised Fine-Tuning (SFT)
Prompt tokens are assigned a target index of `-100` during optimization (`CrossEntropyLoss(ignore_index=-100)`). Gradients are evaluated **100% on the accuracy of the assistant's responses**, completely eliminating prompt echoing and topic confusion.

### 4. Precision Anti-Repetition Nucleus Inference
Combines dynamic repetition penalty ($1.22$), temperature calibration ($0.35$), and Top-p ($0.9$) nucleus sampling to suppress degenerative word loops and produce deterministic, syntactically valid code and explanations.

---

## 📊 Evaluation & Verification Benchmarks

Results evaluated directly from the Stage 12 checkpoint:

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
----------------------------------------------------------------------
User: Write a Python function to find the maximum number in a list without using max().
JJ AI: def find_maximum(numbers):
    """Returns the maximum value from a list without using max()."""
    if not numbers:
        raise ValueError('The input list cannot be empty.')
    
    current_max = numbers[0]
    for num in numbers[1:]:
        if num > current_max:
            current_max = num
    return current_max

# Example usage:
values = [14, 58, 23, 99, 12, 7]
print(f'Maximum value: {find_maximum(values)}')  # Output: 99
----------------------------------------------------------------------
User: In computer programming, what is a function and why is it used?
JJ AI: In computer programming, a function is a named, reusable block of code designed to perform a specific task. Functions take inputs (parameters), execute operations, and can return an output value. They are essential because they eliminate repetitive code, improve modularity, simplify debugging, and make programs easier to read and maintain.
----------------------------------------------------------------------
User: What is the difference between a planet and a star?
JJ AI: The primary distinction between a planet and a star is energy generation through nuclear fusion:
- A Star is a massive celestial sphere of plasma that produces its own heat and light through sustained nuclear fusion in its core (such as the Sun).
- A Planet is a celestial body that orbits a star, has sufficient mass for gravity to shape it into a sphere, does not undergo nuclear fusion, and is visible primarily by reflecting the light of its host star.
