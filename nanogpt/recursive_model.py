"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0) -> torch.Tensor:
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.config.ln_type == "pre": 
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        
        elif self.config.ln_type == "post":
            x = self.ln_1(x + self.attn(x))
            x = self.ln_2(x + self.mlp(x))
        return x

class LatentRec(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fl = Block(config)
        self.fh = Block(config)
    
    def forward(self, x, y, z):
        n = self.config.n_rec_layer
        for _ in range(n):
            print(f"brhu1")
            z = self.fl(x + y + z) # latent reasoning
        print("bruh2")
        y = self.fh(y + z) # refine output answer
        return y, z

@dataclass
class RecursiveGPT2Config:
    block_size: int = 64
    vocab_size: int = 512 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_head: int = 1
    n_prelude_layer: int = 1
    n_rec_layer: int = 2
    n_coda_layer: int = 1
    stop_grad_rec_ratio: float = 0.8
    only_rec_out_norm: bool = False
    ln_type: str = "pre"
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    custom_init: bool = False


class RecursiveGPT2(nn.Module):
    def __init__(self, config: RecursiveGPT2Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            prelude = nn.ModuleList([Block(config) for _ in range(config.n_prelude_layer)]),
            rec_block = nn.ModuleList([Block(config) for _ in range(config.n_rec_layer)]),
            coda = nn.ModuleList([Block(config) for _ in range(config.n_coda_layer)]),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        if self.config.only_rec_out_norm:
            self.rec_out_ln = LayerNorm(config.n_embd, bias=config.bias)
            assert self.config.ln_type != "post", "We don't want Post-LN in the intermediate blocks of the recursive module if `config.only_rec_out_norm == True`"
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        if not self.config.custom_init:
            self.apply(self._init_weights_gpt2)
        else:
            self.apply(self._init_weights_custom)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                if not self.config.custom_init:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                else:
                    in_dim = p.data.shape[-1]
                    std = 1.0/(in_dim**0.5)
                    p.data = trunc_normal_init_(torch.empty_like(p.data), std=std/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights_gpt2(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_weights_custom(self, module):
        if isinstance(module, nn.Linear):
            out_dim, in_dim = module.weight.shape
            module.weight.data = trunc_normal_init_(torch.empty((out_dim, in_dim)), std=1.0/(in_dim**0.5))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight.data = trunc_normal_init_(
                torch.empty((self.config.vocab_size, self.config.n_embd)), 
                std=1.0/(self.config.n_embd**0.5)
            )

    def forward(self, idx: torch.Tensor, targets = None, rec_steps: int = 1):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.prelude:
            x = block(x)
        
        if self.config.stop_grad_rec_ratio > 0 and rec_steps > 1: 
            with torch.no_grad():
                no_grad_steps = int(self.config.stop_grad_rec_ratio * rec_steps)
                for i in range(no_grad_steps):
                    for block in self.transformer.rec_block:
                        x = block(x)
        else:
            no_grad_steps = 0
        
        for j in range(rec_steps - no_grad_steps):
            for block in self.transformer.rec_block:
                x = block(x)
        
        if self.config.only_rec_out_norm:
            x = self.rec_out_ln(x) 
        
        for block in self.transformer.coda:
            x = block(x)
         
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def forward_through_prelude(self, idx: torch.Tensor):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.prelude:
            x = block(x)
        
        return x

@dataclass
class TRMCarry:
    y: torch.Tensor
    z: torch.Tensor


@dataclass
class TRMConfig:
    block_size: int = 64
    vocab_size: int = 512 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 4
    n_head: int = 1
    n_prelude_layer: int = 0    # set to 0 for TRM
    n_rec_layer: int = 6        # this is the n=6 from TRM
    n_latent_rec: int = 3       # number of times latent_recursion is called. equivalent to T=3 from TRM
    n_coda_layer: int = 0       # set to 0 for TRM
    # stop_grad_rec_ratio: float = 0.8  
    # not needed. in TRM, only last latent_recursion skipped no_grad()
    only_rec_out_norm: bool = False
    ln_type: str = "post"       # TRM uses post-LN for contractivity
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    custom_init: bool = False


class TRM(nn.Module):
    def __init__(self, config: RecursiveGPT2Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            prelude = nn.ModuleList([Block(config) for _ in range(config.n_prelude_layer)]),
            # rec_block = nn.ModuleList([Block(config) for _ in range(config.n_rec_layer)]),
            latent_rec = nn.ModuleList([LatentRec(config)] * config.n_latent_rec), # T times latent_recursion calls. 
            # replace each block in rec_block with a latent_recursion step! 
            # n layers are identical and the final one is different.
            coda = nn.ModuleList([Block(config) for _ in range(config.n_coda_layer)]),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # ln_f = LayerNorm(config.n_embd, bias=config.bias), 
            ln_f = nn.Identity(), # not needed since post-LN already enforced.
        ))
        
        if self.config.only_rec_out_norm:
            self.rec_out_ln = LayerNorm(config.n_embd, bias=config.bias)
            assert self.config.ln_type != "post", "We don't want Post-LN in the intermediate blocks of the recursive module if `config.only_rec_out_norm == True`"
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        if not self.config.custom_init:
            self.apply(self._init_weights_gpt2)
        else:
            self.apply(self._init_weights_custom)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                if not self.config.custom_init:
                    torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
                else:
                    in_dim = p.data.shape[-1]
                    std = 1.0/(in_dim**0.5)
                    p.data = trunc_normal_init_(torch.empty_like(p.data), std=std/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights_gpt2(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_weights_custom(self, module):
        if isinstance(module, nn.Linear):
            out_dim, in_dim = module.weight.shape
            module.weight.data = trunc_normal_init_(torch.empty((out_dim, in_dim)), std=1.0/(in_dim**0.5))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            module.weight.data = trunc_normal_init_(
                torch.empty((self.config.vocab_size, self.config.n_embd)), 
                std=1.0/(self.config.n_embd**0.5)
            )

    def forward(self, idx: torch.Tensor, carry: TRMCarry, targets = None, rec_steps: int = 1):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.prelude:
            x = block(x)
        
        # get y_init and z_init
        y, z = carry.z, carry.y

        if rec_steps > 1: 
            with torch.no_grad():
                no_grad_steps = rec_steps - 1
                for i in range(no_grad_steps):
                    for block in self.transformer.latent_rec:
                        y, z = block(x, y, z)
        else:
            no_grad_steps = 0
        
        assert no_grad_steps + 1 == rec_steps

        for j in range(rec_steps - no_grad_steps):
            for block in self.transformer.latent_rec:
                y, z = block(x, y, z)
        
        if self.config.only_rec_out_norm:
            x = self.rec_out_ln(x) 
        
        ##### makes no sense anymore since deep_recursion outputs (y,z) directly fed to lm_head (output_head in TRM algo) to obtain answer instead of reassigning x like in RecursiveGPT2 
        #for block in self.transformer.coda:
        #    x = block(x)
        # 
        #x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(y[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def forward_through_prelude(self, idx: torch.Tensor):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.prelude:
            x = block(x)
        
        return x
