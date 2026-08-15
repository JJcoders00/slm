import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_rope_freqs(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[1], :].to(xq.device).view(1, xq.shape[1], 1, -1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class AnchoredTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, int(dim * 2.67))

    def forward(self, x, freqs_cis, context_anchor=None):
        B, S, D = x.shape
        norm_x = self.norm1(x)
        
        # Inject latent context anchor
        if context_anchor is not None:
            norm_x = norm_x + context_anchor
            
        q = self.q_proj(norm_x).view(B, S, self.n_heads, self.head_dim)
        k = self.k_proj(norm_x).view(B, S, self.n_heads, self.head_dim)
        v = self.v_proj(norm_x).view(B, S, self.n_heads, self.head_dim)
        
        q, k = apply_rotary_emb(q, k, freqs_cis)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        
        h = x + self.out_proj(attn_out)
        return h + self.mlp(self.norm2(h))

class JJComputationalModel(nn.Module):
    def __init__(self, vocab_size=8192, dim=384, n_heads=6, n_layers=4, recurrent_steps=3, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.recurrent_steps = recurrent_steps
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([AnchoredTransformerBlock(dim, n_heads) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight
        
        self.anchor_gate = nn.Linear(dim, dim, bias=False)
        self.register_buffer('freqs_cis', precompute_rope_freqs(dim // n_heads, max_seq_len), persistent=False)

    def forward(self, input_ids, targets=None):
        x = self.embed(input_ids)
        context_anchor = torch.tanh(self.anchor_gate(x.mean(dim=1, keepdim=True)))
        
        for _ in range(self.recurrent_steps):
            for block in self.blocks:
                x = block(x, self.freqs_cis, context_anchor=context_anchor)
                
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=150, temperature=0.3, top_k=20, stop_token_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= 512 else input_ids[:, -512:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if stop_token_id is not None and idx_next.item() == stop_token_id:
                break
        return input_ids
