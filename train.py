"""
Autoresearch Jigsaw training script. Single-GPU, single-file.
Bidirectional transformer classifier trained from scratch on comment text.
Model and optimizer adapted from the upstream nanochat-derived pretraining script.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import load_data, evaluate_predictions

# ---------------------------------------------------------------------------
# Encoder classifier
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    sequence_len: int = 256
    vocab_size: int = 50304
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, attn_mask):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Bidirectional attention; padded keys are masked out
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)
        self.dropout = config.dropout

    def forward(self, x, cos_sin, attn_mask):
        x = x + F.dropout(self.attn(norm(x), cos_sin, attn_mask), self.dropout, self.training)
        x = x + F.dropout(self.mlp(norm(x)), self.dropout, self.training)
        return x


class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.head = nn.Linear(config.n_embd, 1)
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, prior):
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.head.weight, mean=0.0, std=0.001)
        # Start from the training-set base rate instead of p=0.5
        self.head.bias.fill_(math.log(prior / (1 - prior)))
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.wte.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        transformer_matrices = sum(p.numel() for p in self.h.parameters())
        total = wte + head + transformer_matrices
        return {'wte': wte, 'head': head, 'transformer_matrices': transformer_matrices, 'total': total}

    def setup_optimizer(self, head_lr=0.004, embedding_lr=0.05, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95)):
        model_dim = self.config.n_embd
        matrix_params = list(self.h.parameters())
        embedding_params = list(self.wte.parameters())
        head_params = list(self.head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(head_params)
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=head_params, lr=head_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, lengths):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        keep = torch.arange(T, device=idx.device)[None, :] < lengths[:, None]
        attn_mask = keep[:, None, None, :]

        x = self.wte(idx).to(torch.bfloat16)
        x = norm(x)
        x = F.dropout(x, self.config.dropout, self.training)
        for block in self.h:
            x = block(x, cos_sin, attn_mask)
        x = norm(x).float()

        # Mean-pool the real (non-padding) tokens
        pooled = (x * keep.unsqueeze(-1)).sum(dim=1) / lengths[:, None]
        # fp32 head: bf16 logits would tie many predictions and blur the AUC ranking
        with torch.autocast(device_type="cuda", enabled=False):
            logits = self.head(pooled).squeeze(-1)
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Data
MAX_SEQ_LEN = 256       # comments are truncated to their first MAX_SEQ_LEN tokens
LENGTH_BUCKET = 32      # batches are padded to a multiple of this many tokens
SORT_CHUNK_BATCHES = 50 # batches per length-sorted chunk (bigger = less padding, less random)

# Model architecture
DEPTH = 10              # number of transformer layers
MODEL_DIM = 640         # residual stream width
HEAD_DIM = 64           # attention head dimension
DROPOUT = 0.1           # dropout on embeddings and residual branches

# Optimization
EPOCHS = 4              # passes over the training rows (may be fractional)
BATCH_SIZE = 256        # comments per optimizer step
EVAL_BATCH_SIZE = 1024  # comments per inference batch
EMBEDDING_LR = 0.05     # learning rate for token embeddings (Adam)
HEAD_LR = 0.004         # learning rate for the classification head (Adam)
MATRIX_LR = 0.02        # learning rate for matrix parameters (Muon)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.02     # fraction of steps for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of steps for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
LOG_EVERY = 50          # steps between log lines (and NaN checks)

# Artifacts: results/RUN_TAG/EXPERIMENT_ID/
RUN_TAG = os.environ.get("RUN_TAG", "dev")
EXPERIMENT_ID = os.environ.get("EXPERIMENT_ID", time.strftime("%Y%m%d_%H%M%S"))

# ---------------------------------------------------------------------------
# Setup: data, tokenizer, model, optimizer
# ---------------------------------------------------------------------------

t_start = time.time()
sys.stdout.reconfigure(line_buffering=True)  # keep run.log live when redirected
torch.manual_seed(42)
torch.cuda.manual_seed(42)
rng = np.random.default_rng(42)
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 64
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

artifact_dir = Path(__file__).resolve().parent / "results" / RUN_TAG / EXPERIMENT_ID
artifact_dir.mkdir(parents=True, exist_ok=True)
print(f"Artifact dir: {artifact_dir}")

train_df, validation_df, _ = load_data()  # fit on train_df only
print(f"Rows: train={len(train_df):,} validation={len(validation_df):,}")

tokenizer = tiktoken.get_encoding("gpt2")  # fixed pretrained BPE, nothing is fit here
pad_id = tokenizer.n_vocab
vocab_size = ((tokenizer.n_vocab + 1 + 63) // 64) * 64
print(f"Vocab size: {vocab_size:,}")

def tokenize(texts):
    """Returns a padded (rows, MAX_SEQ_LEN) int32 token matrix and the token lengths."""
    tokens = np.full((len(texts), MAX_SEQ_LEN), pad_id, dtype=np.int32)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for start in range(0, len(texts), 100_000):
        ids = tokenizer.encode_ordinary_batch(texts[start:start + 100_000], num_threads=os.cpu_count())
        chunk_lengths = np.array([min(len(t), MAX_SEQ_LEN) for t in ids])
        flat = np.fromiter((tok for t in ids for tok in t[:MAX_SEQ_LEN]), dtype=np.int32, count=chunk_lengths.sum())
        chunk = tokens[start:start + len(ids)]
        chunk[np.arange(MAX_SEQ_LEN)[None, :] < chunk_lengths[:, None]] = flat
        lengths[start:start + len(ids)] = chunk_lengths
    # Empty comments keep one padding token so attention always has a key
    return tokens, np.maximum(lengths, 1)

t0 = time.time()
train_tokens_np, train_lengths_np = tokenize(train_df["comment_text"].tolist())
val_tokens_np, val_lengths_np = tokenize(validation_df["comment_text"].tolist())
print(f"Tokenized in {time.time() - t0:.1f}s | mean train length: {train_lengths_np.mean():.1f} | "
      f"truncated: {100 * (train_lengths_np == MAX_SEQ_LEN).mean():.2f}%")

# The whole tokenized dataset lives on the GPU, so the input pipeline never starves it
train_tokens = torch.from_numpy(train_tokens_np).to(device)
train_lengths = torch.from_numpy(train_lengths_np).to(device)
train_targets = torch.from_numpy(train_df["target"].to_numpy(dtype=np.float32)).to(device)
val_tokens = torch.from_numpy(val_tokens_np).to(device)
val_lengths = torch.from_numpy(val_lengths_np).to(device)
del train_tokens_np, val_tokens_np

config = EncoderConfig(
    sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
    n_layer=DEPTH, n_head=MODEL_DIM // HEAD_DIM, n_embd=MODEL_DIM, dropout=DROPOUT,
)
print(f"Model config: {asdict(config)}")

with torch.device(device):
    raw_model = Classifier(config)
raw_model.init_weights(prior=float(train_df["target"].mean()))

param_counts = raw_model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']

optimizer = raw_model.setup_optimizer(
    head_lr=HEAD_LR,
    embedding_lr=EMBEDDING_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

model = torch.compile(raw_model, dynamic=False)  # one graph per padded length, train and eval

steps_per_epoch = len(train_df) // BATCH_SIZE
total_steps = int(EPOCHS * steps_per_epoch)
print(f"Steps per epoch: {steps_per_epoch:,} | total steps: {total_steps:,}")

def padded_length(lengths):
    return min(MAX_SEQ_LEN, -(-int(lengths.max()) // LENGTH_BUCKET) * LENGTH_BUCKET)

def epoch_batches():
    """Shuffled batches of similar-length comments: shuffle, sort within chunks, shuffle batches."""
    order = rng.permutation(len(train_lengths_np))
    chunks = np.array_split(order, math.ceil(len(order) / (BATCH_SIZE * SORT_CHUNK_BATCHES)))
    order = np.concatenate([c[np.argsort(train_lengths_np[c], kind="stable")] for c in chunks])
    batches = order[:steps_per_epoch * BATCH_SIZE].reshape(steps_per_epoch, BATCH_SIZE)
    return batches[rng.permutation(steps_per_epoch)]

@torch.no_grad()
def predict(tokens, lengths, lengths_np):
    """Probabilities in the given row order."""
    model.eval()
    order = np.argsort(lengths_np, kind="stable")
    probabilities = np.empty(len(order), dtype=np.float64)
    for start in range(0, len(order), EVAL_BATCH_SIZE):
        rows = order[start:start + EVAL_BATCH_SIZE]
        T = padded_length(lengths_np[rows])
        # Repeat the last row so every inference batch has the same shape
        idx = torch.from_numpy(np.pad(rows, (0, EVAL_BATCH_SIZE - len(rows)), mode="edge")).to(device)
        with autocast_ctx:
            logits = model(tokens[idx, :T], lengths[idx])
        probabilities[rows] = torch.sigmoid(logits.double())[:len(rows)].cpu().numpy()
    model.train()
    return probabilities

# Schedules (all based on progress = step / total_steps)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
step = 0
epoch = 0
t_log = time.time()
model.train()

while step < total_steps:
    for rows in epoch_batches():
        if step >= total_steps:
            break
        T = padded_length(train_lengths_np[rows])
        idx = torch.from_numpy(rows).to(device)
        with autocast_ctx:
            logits = model(train_tokens[idx, :T], train_lengths[idx])
        # Soft labels: the target is the fraction of annotators who found the comment toxic
        loss = F.binary_cross_entropy_with_logits(logits, train_targets[idx])
        loss.backward()

        # Progress and schedules
        progress = step / total_steps
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        model.zero_grad(set_to_none=True)

        # Logging
        if step % LOG_EVERY == 0 or step == total_steps - 1:
            train_loss_f = loss.item()

            # Fast fail: abort if loss is exploding or NaN
            if math.isnan(train_loss_f) or train_loss_f > 100:
                print("FAIL")
                exit(1)

            ema_beta = 0.9
            logs = step // LOG_EVERY + 1
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**logs)
            dt = (time.time() - t_log) / (LOG_EVERY if step > 0 else 1)
            t_log = time.time()
            elapsed = time.time() - t_start_training
            remaining = elapsed / (step + 1) * (total_steps - step - 1)
            print(f"step {step:05d} ({100 * progress:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | epoch: {epoch} | elapsed: {elapsed:.0f}s | remaining: {remaining:.0f}s", flush=True)

        # GC management (Python's GC causes ~500ms stalls)
        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

    epoch += 1
    if step < total_steps:
        # Progress report only: nothing is selected or tuned on this number within a run
        epoch_score = evaluate_predictions(predict(val_tokens, val_lengths, val_lengths_np))["score"]
        print(f"epoch {epoch} done | step {step:05d} | val_score: {epoch_score:.6f}", flush=True)

torch.cuda.synchronize()
total_training_time = time.time() - t_start_training

# Final eval
validation_probabilities = predict(val_tokens, val_lengths, val_lengths_np)
metrics = evaluate_predictions(validation_probabilities)
val_score = metrics["score"]

# Artifacts
pd.DataFrame({"id": validation_df["id"], "prediction": validation_probabilities}).to_csv(
    artifact_dir / "validation_predictions.csv", index=False)
torch.save(raw_model.state_dict(), artifact_dir / "checkpoint.pt")
hyperparameters = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple))}
with open(artifact_dir / "config.json", "w") as f:
    json.dump({"hyperparameters": hyperparameters, "model": asdict(config), "tokenizer": "tiktoken/gpt2"}, f, indent=2)
with open(artifact_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Final summary
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_score:        {val_score:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"overall_auc:      {metrics['overall_auc']:.6f}")
print(f"subgroup_auc:     {metrics['power_means']['subgroup']:.6f}")
print(f"bpsn_auc:         {metrics['power_means']['bpsn']:.6f}")
print(f"bnsp_auc:         {metrics['power_means']['bnsp']:.6f}")
print(f"num_steps:        {step}")
print(f"num_epochs:       {EPOCHS}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
