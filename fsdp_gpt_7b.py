import os
import time
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


# ============================================================
# Model / benchmark configuration
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
LEARNING_RATE = 1e-3

DTYPE = torch.bfloat16


# ============================================================
# Model components
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(rms + self.eps)
        return (self.weight.float() * x_fp32).to(input_dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, device=None, dtype=None):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(
            hidden_size,
            3 * hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.out_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
        return self.out_proj(y)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, device=None, dtype=None):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, device=device, dtype=dtype)
        self.attn = CausalSelfAttention(
            hidden_size,
            num_heads,
            device=device,
            dtype=dtype,
        )
        self.mlp_norm = RMSNorm(hidden_size, device=device, dtype=dtype)
        self.mlp = SwiGLUMLP(
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT7B(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.tok_embeddings = nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    HIDDEN_SIZE,
                    NUM_HEADS,
                    INTERMEDIATE_SIZE,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(NUM_LAYERS)
            ]
        )
        self.norm = RMSNorm(HIDDEN_SIZE, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_ids):
        x = self.tok_embeddings(input_ids)

        # Keep activation checkpointing identical in spirit to the DDP stage.
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)

        x = self.norm(x)
        return self.lm_head(x)


# ============================================================
# Helpers
# ============================================================
def init_distributed():
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def count_parameters(model):
    # Count before wrapping so this is the logical full-model parameter count,
    # not the local FSDP shard size.
    return sum(p.numel() for p in model.parameters())


def max_across_ranks(value, device):
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def print_header(rank, world_size, num_params):
    if rank != 0:
        return

    print("=" * 60, flush=True)
    print("FSDP GPT 7B Benchmark", flush=True)
    print("=" * 60, flush=True)
    print(f"World size       : {world_size}", flush=True)
    print(f"Parameters       : {num_params / 1e9:.2f} B", flush=True)
    print(f"Hidden size      : {HIDDEN_SIZE}", flush=True)
    print(f"Layers           : {NUM_LAYERS}", flush=True)
    print(f"Heads            : {NUM_HEADS}", flush=True)
    print(f"Sequence length  : {SEQ_LEN}", flush=True)
    print(f"Batch/GPU        : {BATCH_PER_GPU}", flush=True)
    print(f"Global batch     : {BATCH_PER_GPU * world_size}", flush=True)
    print("Precision        : BF16", flush=True)
    print("Activation ckpt  : Enabled", flush=True)
    print("Optimizer        : SGD", flush=True)
    print("FSDP strategy    : FULL_SHARD", flush=True)
    print("FSDP wrap unit   : TransformerBlock", flush=True)
    print("CPU offload      : Disabled", flush=True)
    print("=" * 60, flush=True)


def main():
    rank, world_size, local_rank, device = init_distributed()

    # Give every rank the same model initialization while keeping benchmark
    # input generation deterministic per rank.
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Build directly in BF16 on the target GPU. This avoids first creating a
    # ~27 GB FP32 copy of the 6.74B model on every rank.
    model = GPT7B(device=device, dtype=DTYPE)
    num_params = count_parameters(model)

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    mixed_precision_policy = MixedPrecision(
        param_dtype=DTYPE,
        reduce_dtype=DTYPE,
        buffer_dtype=DTYPE,
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print_header(rank, world_size, num_params)

    # Synchronize before measurement begins.
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    measured_step_times = []
    total_steps = WARMUP_STEPS + TRAIN_STEPS
    tokens_per_step = BATCH_PER_GPU * SEQ_LEN * world_size

    for step in range(total_steps):
        # Rank-specific synthetic data; the workload shape is identical on all ranks.
        g = torch.Generator(device=device)
        g.manual_seed(100000 + rank * total_steps + step)

        input_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (BATCH_PER_GPU, SEQ_LEN),
            device=device,
            dtype=torch.long,
            generator=g,
        )
        targets = torch.randint(
            0,
            VOCAB_SIZE,
            (BATCH_PER_GPU, SEQ_LEN),
            device=device,
            dtype=torch.long,
            generator=g,
        )

        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        start = time.perf_counter()

        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
        )
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize(device)
        local_step_time = time.perf_counter() - start

        # Distributed step time is determined by the slowest rank.
        step_time = max_across_ranks(local_step_time, device)
        tokens_per_second = tokens_per_step / step_time

        # FSDP memory should be inspected per rank; report the maximum observed
        # peak allocation across all ranks for a conservative cluster figure.
        local_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_mem = max_across_ranks(local_peak_mem, device)

        # Loss values need not be identical because each rank uses different data.
        # Rank 0's loss is sufficient as a workload sanity signal.
        phase = "WARMUP" if step < WARMUP_STEPS else "TRAIN "

        if step >= WARMUP_STEPS:
            measured_step_times.append(step_time)

        if rank == 0:
            print(
                f"{phase} step={step:02d} "
                f"loss={loss.item():.4f} "
                f"time={step_time * 1000:.2f} ms "
                f"tokens/s={tokens_per_second:,.0f} "
                f"max_mem={peak_mem:.2f} GiB",
                flush=True,
            )

    avg_step_time = sum(measured_step_times) / len(measured_step_times)
    global_throughput = tokens_per_step / avg_step_time
    local_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_mem = max_across_ranks(local_peak_mem, device)

    if rank == 0:
        print("", flush=True)
        print("=" * 60, flush=True)
        print("FSDP GPT 7B Result", flush=True)
        print("=" * 60, flush=True)
        print(f"Average step time : {avg_step_time * 1000:.2f} ms", flush=True)
        print(f"Global throughput : {global_throughput:,.0f} tokens/s", flush=True)
        print(f"Tokens / step     : {tokens_per_step:,}", flush=True)
        print(f"Peak GPU memory   : {peak_mem:.2f} GiB", flush=True)
        print("=" * 60, flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
