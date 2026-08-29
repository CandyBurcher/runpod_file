import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint


# ============================================================
# 7B Configuration
# ============================================================

VOCAB_SIZE = 32000
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
NUM_HEADS = 32
INTERMEDIATE_SIZE = 11008

SEQ_LEN = 2048
BATCH_PER_GPU = 1

WARMUP_STEPS = 5
TRAIN_STEPS = 20

DTYPE = torch.bfloat16


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return (self.weight * x).to(dtype=x.dtype)


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = RMSNorm(HIDDEN_SIZE)

        self.qkv = nn.Linear(
            HIDDEN_SIZE,
            3 * HIDDEN_SIZE,
            bias=False
        )

        self.attn_out = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            bias=False
        )

        self.norm2 = RMSNorm(HIDDEN_SIZE)

        # SwiGLU MLP
        self.gate = nn.Linear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            bias=False
        )

        self.up = nn.Linear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            bias=False
        )

        self.down = nn.Linear(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            bias=False
        )

    def forward(self, x):

        B, T, C = x.shape

        # -----------------------
        # Attention
        # -----------------------

        h = self.norm1(x)

        qkv = self.qkv(h)

        q, k, v = qkv.chunk(3, dim=-1)

        head_dim = C // NUM_HEADS

        q = q.view(
            B, T, NUM_HEADS, head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, NUM_HEADS, head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, NUM_HEADS, head_dim
        ).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True
        )

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(B, T, C)

        x = x + self.attn_out(attn)

        # -----------------------
        # SwiGLU
        # -----------------------

        h = self.norm2(x)

        mlp = F.silu(self.gate(h)) * self.up(h)

        x = x + self.down(mlp)

        return x


# ============================================================
# GPT 7B
# ============================================================

class GPT7B(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE
        )

        self.layers = nn.ModuleList([
            TransformerBlock()
            for _ in range(NUM_LAYERS)
        ])

        self.norm = RMSNorm(HIDDEN_SIZE)

        # deliberately NOT weight-tied
        # brings total params close to 6.7B
        self.lm_head = nn.Linear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            bias=False
        )

    def forward(self, idx):

        x = self.embedding(idx)

        # Activation checkpointing
        for layer in self.layers:
            x = checkpoint(
                layer,
                x,
                use_reentrant=False
            )

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# Main
# ============================================================

def main():

    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}"
    )

    torch.manual_seed(1234 + rank)
    torch.cuda.manual_seed(1234 + rank)

    # Enable TF32 where applicable
    torch.backends.cuda.matmul.allow_tf32 = True

    # Build directly on GPU
    # Avoid huge CPU model copies
    with torch.device(device):
        model = GPT7B()

    model = model.to(dtype=DTYPE)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # SGD avoids Adam optimizer-state explosion in pure DDP.
    # --------------------------------------------------------

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-4
    )

    if rank == 0:

        print()
        print("=" * 60)
        print("DDP GPT 7B Benchmark")
        print("=" * 60)

        print(
            f"World size       : {world_size}"
        )

        print(
            f"Parameters       : "
            f"{parameter_count / 1e9:.2f} B"
        )

        print(
            f"Hidden size      : {HIDDEN_SIZE}"
        )

        print(
            f"Layers           : {NUM_LAYERS}"
        )

        print(
            f"Heads            : {NUM_HEADS}"
        )

        print(
            f"Sequence length  : {SEQ_LEN}"
        )

        print(
            f"Batch/GPU        : {BATCH_PER_GPU}"
        )

        print(
            f"Global batch     : "
            f"{BATCH_PER_GPU * world_size}"
        )

        print(
            "Precision        : BF16"
        )

        print(
            "Activation ckpt  : Enabled"
        )

        print(
            "Optimizer        : SGD"
        )

        print("=" * 60)
        print()

    torch.cuda.reset_peak_memory_stats()

    total_steps = (
        WARMUP_STEPS +
        TRAIN_STEPS
    )

    train_times = []

    for step in range(total_steps):

        # ---------------------------------------------
        # Synthetic training data
        # ---------------------------------------------

        tokens = torch.randint(
            0,
            VOCAB_SIZE,
            (
                BATCH_PER_GPU,
                SEQ_LEN + 1
            ),
            device=device
        )

        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        optimizer.zero_grad(
            set_to_none=True
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        # ---------------------------------------------
        # Forward
        # ---------------------------------------------

        logits = model(inputs)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE
            ),
            targets.reshape(-1)
        )

        # ---------------------------------------------
        # Backward
        # DDP gradient AllReduce happens here
        # ---------------------------------------------

        loss.backward()

        optimizer.step()

        torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        # Slowest GPU defines step time
        time_tensor = torch.tensor(
            elapsed,
            device=device
        )

        dist.all_reduce(
            time_tensor,
            op=dist.ReduceOp.MAX
        )

        elapsed = time_tensor.item()

        tokens_per_step = (
            BATCH_PER_GPU
            * SEQ_LEN
            * world_size
        )

        throughput = (
            tokens_per_step
            / elapsed
        )

        max_mem = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

        if rank == 0:

            stage = (
                "WARMUP"
                if step < WARMUP_STEPS
                else "TRAIN "
            )

            print(
                f"{stage} "
                f"step={step:02d} "
                f"loss={loss.item():.4f} "
                f"time={elapsed*1000:.2f} ms "
                f"tokens/s={throughput:,.0f} "
                f"max_mem={max_mem:.2f} GiB",
                flush=True
            )

        if step >= WARMUP_STEPS:
            train_times.append(elapsed)

    # ========================================================
    # Result
    # ========================================================

    avg_time = (
        sum(train_times)
        / len(train_times)
    )

    tokens_per_step = (
        BATCH_PER_GPU
        * SEQ_LEN
        * world_size
    )

    throughput = (
        tokens_per_step
        / avg_time
    )

    if rank == 0:

        print()
        print("=" * 60)
        print("DDP GPT 7B Result")
        print("=" * 60)

        print(
            f"Average step time : "
            f"{avg_time*1000:.2f} ms"
        )

        print(
            f"Global throughput : "
            f"{throughput:,.0f} tokens/s"
        )

        print(
            f"Tokens / step     : "
            f"{tokens_per_step:,}"
        )

        print(
            f"Peak GPU memory   : "
            f"{max_mem:.2f} GiB"
        )

        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
