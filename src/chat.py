import os
import argparse
import torch
from tokenizers import ByteLevelBPETokenizer
from model import JJComputationalModel

def main():
    parser = argparse.ArgumentParser(description="JJ Coders SLM Chat CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/jj_step4_anchored_model.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer", help="Path to tokenizer directory")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running inference on: {device}")

    if not os.path.exists(os.path.join(args.tokenizer_dir, "vocab.json")):
        print(f"Error: Tokenizer files not found at {args.tokenizer_dir}")
        return

    tokenizer = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt")
    )

    model = JJComputationalModel(vocab_size=8192, dim=384, n_heads=6, n_layers=4, recurrent_steps=3, max_seq_len=512).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded checkpoint from: {args.checkpoint} (Loss: {ckpt.get('loss', 'N/A'):.4f})")
    else:
        print("Warning: Checkpoint not found. Running with uninitialized weights.")

    end_id = tokenizer.token_to_id("<|endoftext|>")

    print("\n=== JJ Coders SLM Interactive Chat (Type 'exit' to quit) ===")
    while True:
        try:
            user_prompt = input("\nUser: ").strip()
            if not user_prompt or user_prompt.lower() in ("exit", "quit"):
                break

            formatted_prompt = f"<user> {user_prompt} <bot>"
            input_ids = torch.tensor([tokenizer.encode(formatted_prompt).ids], device=device)
            prompt_len = input_ids.shape[1]

            output_ids = model.generate(input_ids, max_new_tokens=180, temperature=args.temperature, top_k=20, stop_token_id=end_id)
            
            # Slice newly generated tokens
            new_tokens = output_ids[0][prompt_len:]
            response = tokenizer.decode(new_tokens.tolist()).replace("<|endoftext|>", "").strip()
            print(f"JJ AI: {response}")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
