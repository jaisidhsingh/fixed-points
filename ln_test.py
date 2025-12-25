import torch
import torch.nn as nn
import math
from torch.func import jacrev, vmap

def rms(x):
    return torch.sqrt(x.pow(2).mean(dim=-1) + 1e-5)

# -------------------------
# Transformer encoder layer (Post-LN)
# -------------------------
class PostLNTransformerEncoderLayer(nn.Module):
	def __init__(self, d_model, nhead):
		super().__init__()
		self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
		self.ff = nn.Sequential(
			nn.Linear(d_model, 4*d_model),
			nn.ReLU(),
			nn.Linear(4*d_model, d_model),
		)
		self.ln1 = nn.RMSNorm(d_model)
		self.ln2 = nn.RMSNorm(d_model)

	def forward(self, x):
		for i in range(1):
			attn_out, _ = self.attn(x, x, x)
			print(rms(x + attn_out).flatten().mean().item())
			x = self.ln1(x + attn_out)   # post-LN
			ff_out = self.ff(x)
			x = self.ln2(x + ff_out)     # post-LN
		return x


class PreLNTransformerEncoderLayer(nn.Module):
	def __init__(self, d_model, nhead):
		super().__init__()
		self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
		self.ff = nn.Sequential(
			nn.Linear(d_model, 4*d_model),
			nn.ReLU(),
			nn.Linear(4*d_model, d_model),
		)
		self.ln1 = nn.RMSNorm(d_model)
		self.ln2 = nn.RMSNorm(d_model)

	def forward(self, x):
		for i in range(1):
			print(rms(x).flatten().mean().item())
			attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
			x = x + attn_out   # post-LN
			ff_out = self.ff(self.ln2(x))
			x = x + ff_out     # post-LN
		return x

# -------------------------
# Same layer but WITHOUT post-LN
# -------------------------
class NoLNTransformerEncoderLayer(nn.Module):
	def __init__(self, d_model, nhead):
		super().__init__()
		self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
		self.ff = nn.Sequential(
			nn.Linear(d_model, 4*d_model),
			nn.ReLU(),
			nn.Linear(4*d_model, d_model),
		)

	def forward(self, x):
		for i in range(1):
			attn_out, _ = self.attn(x, x, x)
			x = x + attn_out
			x = x + self.ff(x)
		return x


# -------------------------
# Finite-difference Jacobian norm (contractivity test)
# -------------------------
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


def jacobian_op_norm(layer, x):
    def f(inp):
        return layer(inp).reshape(-1)

    J = jacrev(f)(x).reshape(x.numel(), x.numel())
    return torch.linalg.svdvals(J)[0].item()

# -------------------------
# Run test
# -------------------------
torch.manual_seed(0)

B, T, D = 2, 81, 512

device = "mps"
x = torch.randn(B, T, D).to(device)

def rms(x):
    return torch.sqrt(x.pow(2).mean(dim=-1) + 1e-5)

postln = PostLNTransformerEncoderLayer(D, nhead=4).to(device)
noln   = NoLNTransformerEncoderLayer(D, nhead=4).to(device)
preln = PreLNTransformerEncoderLayer(D, nhead=4).to(device)

# postln_norm = finite_diff_jacobian_norm(postln, x)
noln_norm   = finite_diff_jacobian_norm(noln, x)
preln_norm   = finite_diff_jacobian_norm(preln, x)

# print("RMS(x)", rms(x).flatten().mean())
# print("Approx Jacobian operator norm")
# print("Post-LN :", postln_norm)
# print("Pre-LN :", preln_norm)
# print("No LN   :", noln_norm)
