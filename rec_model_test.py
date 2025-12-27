import torch
from nanogpt.recursive_model import RecursiveGPT2, RecursiveGPT2Config



@torch.no_grad()
def finite_diff_jacobian_norm(layer, x, eps=1e-4, n_dirs=1):
    """
    Approximates operator norm via random-direction finite differences:
    ||J|| ≈ max ||f(x+eps*v)-f(x)|| / (eps*||v||)
    """
    y = layer(x)
    max_gain = 0.0
    for _ in range(n_dirs):
        v = torch.randn_like(x)
        v = v / v.norm()
        y_pert = layer(x + eps * v)
        gain = (y_pert - y).norm() / eps
        max_gain = max(max_gain, gain.item())
    return max_gain


def basic_forward_pass_test(device):
    config = RecursiveGPT2Config()
    model = RecursiveGPT2(config).to(device)
    
    idx = torch.tensor([[0, 1, 2]]).long().to(device)
    targets = torch.tensor([[0, 1, 2]]).long().to(device)
    logits, loss = model(idx, targets, rec_steps=6)
    print("Basic forward pass test")
    print("Logits shape:", logits.shape)
    print("NTP loss:", loss.item(), "\n")


@torch.no_grad()
def contractive_recursion_test(device):
    n, p, r, c = 8, 2, 4, 2
    rec_steps = 12
    print("Contractiveness test for recursive block")
    
    for lnt in ["pre", "post", "only_rec_out_norm"]:
        config = RecursiveGPT2Config(
            n_layer=n,
            n_prelude_layer=p,
            n_rec_layer=r,
            n_coda_layer=c,
            ln_type=lnt if lnt != "only_rec_out_norm" else "pre",
            only_rec_out_norm=False if lnt != "only_rec_out_norm" else True
        )
        model = RecursiveGPT2(config).to(device)
        idx = torch.tensor([[0, 1, 2]]).long().to(device)
        
        def recursion(x):
            for i in range(rec_steps):
                for block in model.transformer.rec_block:
                    x = block(x)
            return x
        
        x = model.forward_through_prelude(idx)
        contractiveness = finite_diff_jacobian_norm(recursion, x, n_dirs=1)
        print("LayerNorm type:", lnt)
        print("Contrastiveness:", contractiveness)
    
    print("\n")


def main(device):
    torch.manual_seed(0)
    basic_forward_pass_test(device)
    contractive_recursion_test(device)
    

if __name__ == "__main__":
    device = "mps"
    main(device)
 