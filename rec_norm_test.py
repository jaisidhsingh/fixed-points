import torch
from nanogpt.recursive_model import RecursiveGPT2, RecursiveGPT2Config


def main(device):
    for norm_choice in ['pre', 'post', 'final_block_out_only', 'none']:
        config = RecursiveGPT2Config(
            rec_norm_kind=norm_choice # choices: 'pre' | 'post' | 'final_block_out_only' | 'none'
        )
        model = RecursiveGPT2(config).to(device)
        
        idx = torch.tensor([[0, 1, 2]]).long().to(device)
        targets = torch.tensor([[0, 1, 2]]).long().to(device)
        logits, loss = model(idx, targets, rec_steps=6)
        print(logits.shape)
        print(loss.item())

if __name__ == "__main__":
    device = "cuda"
    main(device)
     
