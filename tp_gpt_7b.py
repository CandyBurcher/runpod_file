#!/usr/bin/env python3
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.checkpoint import checkpoint


# ============================================================
# GPT-7B benchmark configuration
# ============================================================

VOCAB_SIZE = 32000
HIDDEN_SIZE = 4096
NUM_LAYERS = 32
NUM_HEADS = 32
INTERMEDIATE_SIZE = 11008

SEQ_LEN = 2048
MICRO_BATCH_SIZE = 1

WARMUP_STEPS = 5
TRAIN_STEPS = 20

DTYPE = torch.bfloat16
LR = 1e-3

USE_ACTIVATION_CHECKPOINT = True


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(-1, keepdim=True)
        x_fp32 = x_fp32 * torch.rsqrt(rms + self.eps)
        return (self.weight.float() * x_fp32).to(input_dtype)


# ============================================================
# Tensor-parallel Attention
# ============================================================

class Attention(nn.Module):
    def __init__(self, tp_size):
        super().__init__()

        if NUM_HEADS % tp_size != 0:
            raise ValueError(
                f"NUM_HEADS={NUM_HEADS} must be divisible by TP size={tp_size}"
            )

        self.tp_size = tp_size
        self.local_heads = NUM_HEADS // tp_size
        self.head_dim = HIDDEN_SIZE // NUM_HEADS
        self.local_hidden = HIDDEN_SIZE // tp_size

        self.qkv = nn.Linear(
            HIDDEN_SIZE,
            3 * HIDDEN_SIZE,
            bias=False,
        )

        self.proj = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            bias=False,
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        # ColwiseParallel output:
        # [B, T, 3 * HIDDEN_SIZE / TP]
        qkv = self.qkv(x)

        qkv = qkv.view(
            bsz,
            seqlen,
            3,
            self.local_heads,
            self.head_dim,
        )

        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )

        # Local tensor-parallel hidden shard:
        # [B, T, HIDDEN_SIZE / TP]
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(bsz, seqlen, self.local_hidden)
        )

        # RowwiseParallel reduces partial outputs across TP ranks
        # and returns replicated [B, T, HIDDEN_SIZE].
        return self.proj(attn)


# ============================================================
# Tensor-parallel SwiGLU MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self, tp_size):
        super().__init__()

        if INTERMEDIATE_SIZE % tp_size != 0:
            raise ValueError(
                f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} "
                f"must be divisible by TP size={tp_size}"
            )

        self.local_intermediate = INTERMEDIATE_SIZE // tp_size

        self.gate_up = nn.Linear(
            HIDDEN_SIZE,
            2 * INTERMEDIATE_SIZE,
            bias=False,
        )

        self.down = nn.Linear(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            bias=False,
        )

    def forward(self, x):
        # ColwiseParallel output:
        # [B, T, 2 * INTERMEDIATE_SIZE / TP]
        gate_up = self.gate_up(x)

        gate, up = gate_up.split(
            self.local_intermediate,
            dim=-1,
        )

        x = F.silu(gate) * up

        # RowwiseParallel performs the required reduction.
        return self.down(x)


# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, tp_size):
        super().__init__()

        self.attn_norm = RMSNorm(HIDDEN_SIZE)
        self.attn = Attention(tp_size)

        self.ffn_norm = RMSNorm(HIDDEN_SIZE)
        self.mlp = MLP(tp_size)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


# ============================================================
# GPT model
# ============================================================

class GPT7B(nn.Module):
    def __init__(self, tp_size):
        super().__init__()

        self.tok_embeddings = nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(tp_size)
                for _ in range(NUM_LAYERS)
            ]
        )

        self.norm = RMSNorm(HIDDEN_SIZE)

        # Intentionally untied, matching the previous 6.74B model.
        self.output = nn.Linear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            bias=False,
        )

    def forward(self, input_ids):
        x = self.tok_embeddings(input_ids)

        for layer in self.layers:
            if USE_ACTIVATION_CHECKPOINT and self.training:
                x = checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x)

        x = self.norm(x)
        logits = self.output(x)
        return logits


# ============================================================
# Apply native PyTorch Tensor Parallel
# ============================================================

def apply_tensor_parallel(model, tp_mesh):
    # Shard embedding and LM head too, while exposing replicated
    # activations/logits to keep the benchmark/loss simple.
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
            "output": ColwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Replicate(),
            ),
        },
    )

    for layer in model.layers:
        parallelize_module(
            layer,
            tp_mesh,
            {
                "attn.qkv": ColwiseParallel(),
                "attn.proj": RowwiseParallel(),
                "mlp.gate_up": ColwiseParallel(),
                "mlp.down": RowwiseParallel(),
            },
        )

    return model


# ============================================================
# Helpers
# ============================================================

def reduce_max_step_time(step_time, device):
    value = torch.tensor(
        step_time,
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def reduce_max_memory(memory_bytes, device):
    value = torch.tensor(
        float(memory_bytes),
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


# ============================================================
# Main
# ============================================================

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
    )

    # Stage 4 = pure Tensor Parallel:
    # WORLD_SIZE is the TP degree.
    tp_size = world_size

    if tp_size not in (1, 2, 4, 8):
        raise ValueError(
            f"This script expects TP size 1/2/4/8, got {tp_size}. "
            "TP+DP hybrid parallelism will use a separate script."
        )

    if NUM_HEADS % tp_size != 0:
        raise ValueError(
            f"NUM_HEADS={NUM_HEADS} must be divisible by TP size={tp_size}"
        )

    if HIDDEN_SIZE % tp_size != 0:
        raise ValueError(
            f"HIDDEN_SIZE={HIDDEN_SIZE} must be divisible by TP size={tp_size}"
        )

    if INTERMEDIATE_SIZE % tp_size != 0:
        raise ValueError(
            f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} "
            f"must be divisible by TP size={tp_size}"
        )

    # TP ranks must start from the same logical weights.
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    model = GPT7B(tp_size).to(
        device=device,
        dtype=DTYPE,
    )

    global_param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    tp_mesh = init_device_mesh(
        "cuda",
        (tp_size,),
        mesh_dim_names=("tp",),
    )

    model = apply_tensor_parallel(
        model,
        tp_mesh,
    )

    # After TP sharding, p.numel() reflects the local DTensor shard.
    local_param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        foreach=False,
    )

    # All TP ranks cooperate on the SAME micro-batch.
    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (MICRO_BATCH_SIZE, SEQ_LEN),
        device=device,
        dtype=torch.long,
    )

    labels = torch.randint(
        0,
        VOCAB_SIZE,
        (MICRO_BATCH_SIZE, SEQ_LEN),
        device=device,
        dtype=torch.long,
    )

    model.train()

    dist.barrier()
    torch.cuda.synchronize()

    # Exclude construction/sharding memory from measured peak memory.
    torch.cuda.reset_peak_memory_stats(device)

    if rank == 0:
        print("=" * 60)
        print("TP GPT 7B Benchmark")
        print("=" * 60)
        print(f"World size       : {world_size}")
        print(f"TP size          : {tp_size}")
        print(f"Parameters       : {global_param_count / 1e9:.2f} B")
        print(f"Local parameters : {local_param_count / 1e9:.2f} B/rank")
        print(f"Hidden size      : {HIDDEN_SIZE}")
        print(f"Layers           : {NUM_LAYERS}")
        print(f"Heads            : {NUM_HEADS}")
        print(f"Heads / TP rank  : {NUM_HEADS // tp_size}")
        print(f"Sequence length  : {SEQ_LEN}")
        print(f"Micro batch      : {MICRO_BATCH_SIZE}")
        print(f"TP group batch   : {MICRO_BATCH_SIZE}")
        print("Precision        : BF16")
        print(
            "Activation ckpt  : "
            + ("Enabled" if USE_ACTIVATION_CHECKPOINT else "Disabled")
        )
        print("Optimizer        : SGD")
        print("TP implementation: PyTorch DTensor native TP")
        print("QKV / gate_up    : ColwiseParallel")
        print("Proj / down      : RowwiseParallel")
        print("=" * 60)

    # IMPORTANT:
    # Pure TP does NOT multiply samples/tokens by TP size.
    tokens_per_step = MICRO_BATCH_SIZE * SEQ_LEN

    measured_times = []
    total_steps = WARMUP_STEPS + TRAIN_STEPS

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)

        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()

        logits = model(input_ids)

        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            labels.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        local_step_time = time.perf_counter() - start
        step_time = reduce_max_step_time(
            local_step_time,
            device,
        )

        local_peak_mem = torch.cuda.max_memory_allocated(device)
        peak_mem = reduce_max_memory(
            local_peak_mem,
            device,
        )

        tokens_per_second = tokens_per_step / step_time

        if step >= WARMUP_STEPS:
            measured_times.append(step_time)

        if rank == 0:
            phase = (
                "WARMUP"
                if step < WARMUP_STEPS
                else "TRAIN "
            )

            print(
                f"{phase} step={step:02d} "
                f"loss={loss.item():.4f} "
                f"time={step_time * 1000:.2f} ms "
                f"tokens/s={tokens_per_second:,.0f} "
                f"max_mem={peak_mem / (1024 ** 3):.2f} GiB",
                flush=True,
            )

    avg_step_time = sum(measured_times) / len(measured_times)
    throughput = tokens_per_step / avg_step_time

    local_peak_mem = torch.cuda.max_memory_allocated(device)
    peak_mem = reduce_max_memory(
        local_peak_mem,
        device,
    )

    if rank == 0:
        print()
        print("=" * 60)
        print("TP GPT 7B Result")
        print("=" * 60)
        print(f"TP size           : {tp_size}")
        print(f"Average step time : {avg_step_time * 1000:.2f} ms")
        print(f"Global throughput : {throughput:,.0f} tokens/s")
        print(f"Tokens / step     : {tokens_per_step:,}")
        print(f"Peak GPU memory   : {peak_mem / (1024 ** 3):.2f} GiB")
        print("=" * 60)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
