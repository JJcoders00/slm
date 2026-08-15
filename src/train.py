import os
import argparse
import random
import torch
import torch.nn as nn
from model import JJComputationalModel

def get_sft_batch(samples, batch_size=16, pad_id=0, device="cpu"):
    batch = random.sample(samples, batch_size)
    max_len = max(len(s[0]) for s in batch)
    
    x_padded, y_padded = [], []
    for x, y in batch:
        pad_len = max_len - len(x)
        x_padded.append(x + [pad_id] * pad_len)
        y_padded.append(y + [-100] * pad_len)
        
    return torch.tensor(x_padded, dtype=torch.long, device=device), torch.tensor(y_padded, dtype=torch.long, device=device)

def main():
    parser = argparse.ArgumentParser(description="JJ Coders SLM Training Engine")
    parser.add_argument("--steps", type=int, default=3000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    model = JJComputationalModel(vocab_size=8192, dim=384, n_heads=6, n_layers=4, recurrent_steps=3, max_seq_len=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    ckpt_path = os.path.join(args.save_dir, "jj_step4_anchored_model.pt")
    start_step = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_step = ckpt['step'] + 1
        print(f"Resumed from step {start_step} (Saved Loss: {ckpt['loss']:.4f})")

    print(f"Training initialized. Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

if __name__ == "__main__":
    main()
