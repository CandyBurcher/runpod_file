#!/usr/bin/env python3
import argparse
import os
import time

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
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
BATCH_PER_GPU = 1

WARMUP_STEPS = 5
TRAIN_STEPS = 20

LR = 1e-3
DTYPE = torch.bfloat16

USE_ACTIVATION_CHECKPOINT = True


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, dtype=DTYPE)
        )
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x_fp32 = x.float()

        rms = x_fp32.pow(2).mean(
            dim=-1,
            keepdim=True,
        )

        x_fp32 = x_fp32 * torch.rsqrt(
            rms + self.eps
        )

        return (
            self.weight.float() * x_fp32
        ).to(input_dtype)


# ============================================================
# Attention
# ============================================================

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.head_dim = (
            HIDDEN_SIZE // NUM_HEADS
        )

        self.qkv = nn.Linear(
            HIDDEN_SIZE,
            3 * HIDDEN_SIZE,
            bias=False,
            dtype=DTYPE,
        )

        self.proj = nn.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            bias=False,
            dtype=DTYPE,
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        qkv = self.qkv(x)

        qkv = qkv.view(
            bsz,
            seqlen,
            3,
            NUM_HEADS,
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

        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(
                bsz,
                seqlen,
                HIDDEN_SIZE,
            )
        )

        return self.proj(attn)


# ============================================================
# SwiGLU MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.gate_up = nn.Linear(
            HIDDEN_SIZE,
            2 * INTERMEDIATE_SIZE,
            bias=False,
            dtype=DTYPE,
        )

        self.down = nn.Linear(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            bias=False,
            dtype=DTYPE,
        )

    def forward(self, x):
        gate_up = self.gate_up(x)

        gate, up = gate_up.chunk(
            2,
            dim=-1,
        )

        x = F.silu(gate) * up

        return self.down(x)


# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn_norm = RMSNorm(
            HIDDEN_SIZE
        )

        self.attn = Attention()

        self.ffn_norm = RMSNorm(
            HIDDEN_SIZE
        )

        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(
            self.attn_norm(x)
        )

        x = x + self.mlp(
            self.ffn_norm(x)
        )

        return x


# ============================================================
# GPT model
# ============================================================

class GPT7B(nn.Module):
    def __init__(self):
        super().__init__()

        self.tok_embeddings = nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            dtype=DTYPE,
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        self.norm = RMSNorm(
            HIDDEN_SIZE
        )

        # Keep output untied to match the previous
        # 6.74B DDP/FSDP/TP benchmarks.
        self.output = nn.Linear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            bias=False,
            dtype=DTYPE,
        )

    def forward(self, input_ids):
        x = self.tok_embeddings(
            input_ids
        )

        for layer in self.layers:
            if (
                USE_ACTIVATION_CHECKPOINT
                and self.training
            ):
                x = checkpoint(
                    layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer(x)

        x = self.norm(x)

        return self.output(x)


# ============================================================
# Helpers
# ============================================================

def logical_parameter_count(model):
    total = 0

    for p in model.parameters():
        # Under deepspeed.zero.Init(), ZeRO-3 parameters
        # expose ds_numel as the logical/global size.
        total += int(
            getattr(
                p,
                "ds_numel",
                p.numel(),
            )
        )

    return total


def world_max(value, device):
    t = torch.tensor(
        float(value),
        device=device,
        dtype=torch.float64,
    )

    dist.all_reduce(
        t,
        op=dist.ReduceOp.MAX,
    )

    return t.item()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
    )

    parser = deepspeed.add_config_arguments(
        parser
    )

    args = parser.parse_args()

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    torch.cuda.set_device(
        local_rank
    )

    device = torch.device(
        "cuda",
        local_rank,
    )

    # DeepSpeed initializes torch.distributed/NCCL.
    deepspeed.init_distributed(
        dist_backend="nccl"
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # This benchmark is intended for either:
    #   8 GPUs / 1 node
    #   16 GPUs / 2 nodes
    if world_size not in (8, 16):
        raise RuntimeError(
            "This benchmark expects WORLD_SIZE "
            f"8 or 16, but got {world_size}"
        )

    # --------------------------------------------------------
    # Construct the model directly under ZeRO-3 init.
    #
    # This avoids requiring every rank to materialize a full
    # unsharded 6.74B model before DeepSpeed partitions it.
    # --------------------------------------------------------

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    with deepspeed.zero.Init(
        config_dict_or_path=args.deepspeed_config,
    ):
        model = GPT7B()

    param_count = logical_parameter_count(
        model
    )

    # Keep SGD for comparability with the previous
    # DDP/FSDP benchmarks.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        foreach=False,
    )

    model_engine, optimizer, _, _ = (
        deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model.parameters(),
        )
    )

    # --------------------------------------------------------
    # Each DP rank gets its own micro-batch.
    # --------------------------------------------------------

    data_seed = 2026 + rank

    torch.manual_seed(
        data_seed
    )
    torch.cuda.manual_seed(
        data_seed
    )

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (
            BATCH_PER_GPU,
            SEQ_LEN,
        ),
        device=device,
        dtype=torch.long,
    )

    labels = torch.randint(
        0,
        VOCAB_SIZE,
        (
            BATCH_PER_GPU,
            SEQ_LEN,
        ),
        device=device,
        dtype=torch.long,
    )

    model_engine.train()

    dist.barrier()
    torch.cuda.synchronize()

    # Exclude model construction / ZeRO initialization
    # from the benchmark memory figure.
    torch.cuda.reset_peak_memory_stats(
        device
    )

    global_batch = (
        BATCH_PER_GPU
        * world_size
    )

    tokens_per_step = (
        global_batch
        * SEQ_LEN
    )

    if rank == 0:
        print(
            "=" * 60
        )
        print(
            "DeepSpeed ZeRO-3 GPT 7B Benchmark"
        )
        print(
            "=" * 60
        )
        print(
            f"World size       : "
            f"{world_size}"
        )
        print(
            f"Parameters       : "
            f"{param_count / 1e9:.2f} B"
        )
        print(
            f"Hidden size      : "
            f"{HIDDEN_SIZE}"
        )
        print(
            f"Layers           : "
            f"{NUM_LAYERS}"
        )
        print(
            f"Heads            : "
            f"{NUM_HEADS}"
        )
        print(
            f"Intermediate     : "
            f"{INTERMEDIATE_SIZE}"
        )
        print(
            f"Sequence length  : "
            f"{SEQ_LEN}"
        )
        print(
            f"Batch / GPU      : "
            f"{BATCH_PER_GPU}"
        )
        print(
            f"Global batch     : "
            f"{global_batch}"
        )
        print(
            "Precision        : BF16"
        )
        print(
            "Activation ckpt  : "
            + (
                "Enabled"
                if USE_ACTIVATION_CHECKPOINT
                else "Disabled"
            )
        )
        print(
            "Optimizer        : SGD"
        )
        print(
            "ZeRO stage       : 3"
        )
        print(
            "CPU offload      : Disabled"
        )
        print(
            "NVMe offload     : Disabled"
        )
        print(
            "=" * 60
        )

    measured_times = []

    total_steps = (
        WARMUP_STEPS
        + TRAIN_STEPS
    )

    for step in range(total_steps):
        model_engine.zero_grad()

        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()

        logits = model_engine(
            input_ids
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            labels.reshape(-1),
        )

        model_engine.backward(
            loss
        )

        model_engine.step()

        torch.cuda.synchronize()

        local_step_time = (
            time.perf_counter()
            - start
        )

        step_time = world_max(
            local_step_time,
            device,
        )

        local_peak_mem = (
            torch.cuda.max_memory_allocated(
                device
            )
        )

        peak_mem = world_max(
            local_peak_mem,
            device,
        )

        tokens_per_second = (
            tokens_per_step
            / step_time
        )

        if step >= WARMUP_STEPS:
            measured_times.append(
                step_time
            )

        if rank == 0:
            phase = (
                "WARMUP"
                if step < WARMUP_STEPS
                else "TRAIN "
            )

            print(
                f"{phase} "
                f"step={step:02d} "
                f"loss={loss.item():.4f} "
                f"time={step_time * 1000:.2f} ms "
                f"tokens/s={tokens_per_second:,.0f} "
                f"max_mem={peak_mem / (1024 ** 3):.2f} GiB",
                flush=True,
            )

    avg_step_time = (
        sum(measured_times)
        / len(measured_times)
    )

    throughput = (
        tokens_per_step
        / avg_step_time
    )

    local_peak_mem = (
        torch.cuda.max_memory_allocated(
            device
        )
    )

    peak_mem = world_max(
        local_peak_mem,
        device,
    )

    if rank == 0:
        print()
        print(
            "=" * 60
        )
        print(
            "DeepSpeed ZeRO-3 GPT 7B Result"
        )
        print(
            "=" * 60
        )
        print(
            f"World size        : "
            f"{world_size}"
        )
        print(
            f"Average step time : "
            f"{avg_step_time * 1000:.2f} ms"
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
            f"{peak_mem / (1024 ** 3):.2f} GiB"
        )
        print(
            "=" * 60
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
