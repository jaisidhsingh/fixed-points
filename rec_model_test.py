from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch.func import jacrev, vmap
from nanogpt.recursive_model import RecursiveGPT2, RecursiveGPT2Config


def jacobian_op_norm(layer, x):
    def f(inp):
        return layer(inp).reshape(-1)

    J = jacrev(f)(x).reshape(x.numel(), x.numel())
    return torch.linalg.svdvals(J)[0].item()


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
def contractive_recursion_test(device, rec_steps=12):
    fdiff_n_dirs = 100
    n, p, r, c = 8, 2, 4, 2
    n_embd = 128
    print()
    print("Contractiveness test for recursive block. rec_steps=", rec_steps, "  n_embd=", n_embd)
    
    ln_types = ["pre", "post", "only_rec_out_norm"]
    lnt_contractiveness = {}
    for lnt in ln_types:
        config = RecursiveGPT2Config(
            n_embd=n_embd,
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
        
        class TmpRec(torch.nn.Module):
            def __init__(self, f):
                super().__init__()
                self.f = f
            
            def forward(self, x):
                for i in range(rec_steps):
                    for block in self.f:
                        x = block(x)
                return x
        
        x = model.forward_through_prelude(idx)
        
        f = TmpRec(model.transformer.rec_block)
        contractiveness_fdiff = finite_diff_jacobian_norm(f, x, n_dirs=fdiff_n_dirs)
        contractiveness_jnorm  = jacobian_op_norm(f, x)
        print("LayerNorm type:", lnt)
        print(f"Contractiveness (finite_diff_jacobian_norm, n_dirs={fdiff_n_dirs}) :", contractiveness_fdiff)
        print(f"Contractiveness (jacobian_op_norm)                    :", contractiveness_jnorm)
        print()
        lnt_contractiveness[lnt] = contractiveness_jnorm

    print("\n")
    return lnt_contractiveness


def contractive_recursion_test_plots(device, savepath=f"contractive_recursion_test_for_different_R.png", display=False):
    print()
    print(f"running experiment across different number of rec_steps...")
    x_rec_steps = []
    y_post = []
    y_pre = []
    y_onlyout = []
    for rec_steps in range(1, 12):
        x_rec_steps.append(rec_steps)
        lnt_j_op_norm = contractive_recursion_test(device, rec_steps)
        y_post.append(lnt_j_op_norm["post"])
        y_pre.append(lnt_j_op_norm["pre"])
        y_onlyout.append(lnt_j_op_norm["only_rec_out_norm"])
    plt.plot(x_rec_steps, y_post, label="post")
    plt.plot(x_rec_steps, y_pre, label="pre")
    plt.plot(x_rec_steps, y_onlyout, label="only_rec_out_norm")
    plt.legend()
    plt.xlabel("rec_steps")
    plt.ylabel("contractiveness (jacobian_op_norm)")
    plt.savefig(savepath)
    if display:
        plt.show()
    plt.clf()

def main(device):
    torch.manual_seed(0)
    # basic_forward_pass_test(device)
    contractive_recursion_test(device)
    # contractive_recursion_test_plots(device, display=True)


if __name__ == "__main__":
    device = "cpu"
    main(device)
 
