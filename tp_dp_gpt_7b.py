#!/usr/bin/env python3
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.checkpoint import checkpoint


# ============================================================
# Fixed hybrid-parallel topology
#
# 2 nodes x 8 GPUs/node = 16 GPUs
#
# Node-0: ranks  0..7  -> TP replica 0
# Node-1: ranks  8..15 -> TP replica 1
#
# TP_SIZE = 8
# DP_SIZE = 2
#
# TP communication stays inside each node.
# DP synchronization pairs identical TP coordinates:
#   rank 0 <-> rank 8
#   rank 1 <-> rank 9
#   ...
#   rank 7 <-> rank 15
# ============================================================

TP_SIZE = 8
DP_SIZE = 2
EXPECTED_WORLD_SIZE = TP_SIZE * DP_SIZE


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
# Tensor-parallel Attention
#
# qkv:
#   ColwiseParallel
#   hidden -> 3 * hidden / TP
#
# proj:
#   RowwiseParallel
#   hidden / TP -> hidden
#
# With TP=8:
#   32 attention heads / 8 = 4 heads per TP rank
# ============================================================

class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        if NUM_HEADS % TP_SIZE != 0:
            raise ValueError(
                f"NUM_HEADS={NUM_HEADS} must be divisible "
                f"by TP_SIZE={TP_SIZE}"
            )

        self.local_heads = NUM_HEADS // TP_SIZE
        self.head_dim = HIDDEN_SIZE // NUM_HEADS
        self.local_hidden = HIDDEN_SIZE // TP_SIZE

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

        # After ColwiseParallel:
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

        # Local TP shard:
        # [B, T, HIDDEN_SIZE / TP]
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .view(
                bsz,
                seqlen,
                self.local_hidden,
            )
        )

        # RowwiseParallel performs the TP reduction and
        # returns replicated [B, T, HIDDEN_SIZE].
        return self.proj(attn)


# ============================================================
# Tensor-parallel SwiGLU MLP
#
# gate_up:
#   ColwiseParallel
#   hidden -> 2 * intermediate / TP
#
# down:
#   RowwiseParallel
#   intermediate / TP -> hidden
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        if INTERMEDIATE_SIZE % TP_SIZE != 0:
            raise ValueError(
                f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} "
                f"must be divisible by TP_SIZE={TP_SIZE}"
            )

        self.local_intermediate = (
            INTERMEDIATE_SIZE // TP_SIZE
        )

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
        # After ColwiseParallel:
        # [B, T, 2 * INTERMEDIATE_SIZE / TP]
        gate_up = self.gate_up(x)

        gate, up = gate_up.split(
            self.local_intermediate,
            dim=-1,
        )

        x = F.silu(gate) * up

        # RowwiseParallel performs TP reduction.
        return self.down(x)


# ============================================================
# Transformer block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn_norm = RMSNorm(HIDDEN_SIZE)
        self.attn = Attention()

        self.ffn_norm = RMSNorm(HIDDEN_SIZE)
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
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(NUM_LAYERS)
            ]
        )

        self.norm = RMSNorm(HIDDEN_SIZE)

        # Intentionally untied, matching the previous
        # DDP/FSDP/TP 6.74B benchmark.
        self.output = nn.Linear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            bias=False,
        )

    def forward(self, input_ids):
        x = self.tok_embeddings(input_ids)

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
        logits = self.output(x)

        return logits


# ============================================================
# Tensor Parallel plan
# ============================================================

def apply_tensor_parallel(
    model,
    tp_mesh,
):
    # Embedding + LM head
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

    # Transformer blocks
    for layer in model.layers:
        parallelize_module(
            layer,
            tp_mesh,
            {
                "attn.qkv":
                    ColwiseParallel(),

                "attn.proj":
                    RowwiseParallel(),

                "mlp.gate_up":
                    ColwiseParallel(),

                "mlp.down":
                    RowwiseParallel(),
            },
        )

    return model


# ============================================================
# Helpers
# ============================================================

def local_tensor(t):
    """
    Return the physical local tensor.

    DTensor -> local shard
    Tensor  -> itself
    """
    if isinstance(t, DTensor):
        return t.to_local()

    return t


def local_parameter_count(model):
    total = 0

    for p in model.parameters():
        if isinstance(p, DTensor):
            total += p.to_local().numel()
        else:
            total += p.numel()

    return total


def sync_dp_gradients(
    model,
    dp_group,
):
    """
    Data-parallel gradient synchronization.

    Each TP shard is synchronized only with the
    corresponding TP shard in the other DP replica.

    Example:
        rank 0 <-> rank 8
        rank 1 <-> rank 9
        ...
        rank 7 <-> rank 15

    This first hybrid benchmark intentionally uses
    synchronous per-parameter DP AllReduce instead
    of DDP gradient buckets, so the communication
    behavior is explicit and easy to inspect.
    """
    for p in model.parameters():
        if p.grad is None:
            continue

        grad = local_tensor(p.grad)

        dist.all_reduce(
            grad,
            op=dist.ReduceOp.SUM,
            group=dp_group,
        )

        grad.div_(DP_SIZE)


def reduce_world_max(
    value,
    device,
):
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
    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    rank = int(
        os.environ["RANK"]
    )

    world_size = int(
        os.environ["WORLD_SIZE"]
    )

    local_world_size = int(
        os.environ.get(
            "LOCAL_WORLD_SIZE",
            TP_SIZE,
        )
    )

    if world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            f"This benchmark requires exactly "
            f"{EXPECTED_WORLD_SIZE} ranks "
            f"(TP={TP_SIZE} x DP={DP_SIZE}), "
            f"but WORLD_SIZE={world_size}"
        )

    if local_world_size != TP_SIZE:
        raise RuntimeError(
            f"This benchmark expects "
            f"{TP_SIZE} GPUs/processes per node, "
            f"but LOCAL_WORLD_SIZE={local_world_size}"
        )

    torch.cuda.set_device(local_rank)

    device = torch.device(
        "cuda",
        local_rank,
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
    )

    # Global rank layout is expected to be:
    #
    # dp=0: ranks 0..7
    # dp=1: ranks 8..15
    #
    # This matches torchrun with:
    #   2 nodes
    #   8 processes/node
    #
    dp_rank = rank // TP_SIZE
    tp_rank = rank % TP_SIZE

    # --------------------------------------------------------
    # 2-D DeviceMesh
    #
    # Shape:
    #   [DP, TP] = [2, 8]
    #
    # Rows are TP groups:
    #   ranks 0..7
    #   ranks 8..15
    #
    # Columns are DP groups:
    #   [0,8], [1,9], ..., [7,15]
    # --------------------------------------------------------

    mesh = init_device_mesh(
        "cuda",
        (DP_SIZE, TP_SIZE),
        mesh_dim_names=(
            "dp",
            "tp",
        ),
    )

    tp_mesh = mesh["tp"]
    dp_mesh = mesh["dp"]

    dp_group = dp_mesh.get_group()

    # --------------------------------------------------------
    # Identical logical model initialization on all ranks
    # --------------------------------------------------------

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    model = GPT7B().to(
        device=device,
        dtype=DTYPE,
    )

    global_param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    model = apply_tensor_parallel(
        model,
        tp_mesh,
    )

    local_param_count = (
        local_parameter_count(model)
    )

    # foreach=False is required because this model
    # contains a mixture of DTensor parameters and
    # ordinary replicated Tensor parameters.
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        foreach=False,
    )

    # --------------------------------------------------------
    # Different data for different DP replicas.
    #
    # All 8 ranks inside the SAME TP replica must see
    # identical input/labels.
    #
    # dp_rank 0 -> seed 2026
    # dp_rank 1 -> seed 2027
    # --------------------------------------------------------

    data_seed = 2026 + dp_rank

    torch.manual_seed(data_seed)
    torch.cuda.manual_seed(data_seed)

    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (
            MICRO_BATCH_SIZE,
            SEQ_LEN,
        ),
        device=device,
        dtype=torch.long,
    )

    labels = torch.randint(
        0,
        VOCAB_SIZE,
        (
            MICRO_BATCH_SIZE,
            SEQ_LEN,
        ),
        device=device,
        dtype=torch.long,
    )

    model.train()

    dist.barrier()
    torch.cuda.synchronize()

    # Measure training memory only, excluding model
    # construction and TP sharding.
    torch.cuda.reset_peak_memory_stats(
        device
    )

    if rank == 0:
        print(
            "=" * 60
        )
        print(
            "TP + DP GPT 7B Benchmark"
        )
        print(
            "=" * 60
        )
        print(
            f"World size       : "
            f"{world_size}"
        )
        print(
            f"TP size          : "
            f"{TP_SIZE}"
        )
        print(
            f"DP size          : "
            f"{DP_SIZE}"
        )
        print(
            f"Parameters       : "
            f"{global_param_count / 1e9:.2f} B"
        )
        print(
            f"Local parameters : "
            f"{local_param_count / 1e9:.2f} B/rank"
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
            f"Heads / TP rank  : "
            f"{NUM_HEADS // TP_SIZE}"
        )
        print(
            f"Sequence length  : "
            f"{SEQ_LEN}"
        )
        print(
            f"Batch / DP repl. : "
            f"{MICRO_BATCH_SIZE}"
        )
        print(
            f"Global batch     : "
            f"{MICRO_BATCH_SIZE * DP_SIZE}"
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
            "TP implementation: "
            "PyTorch DTensor native TP"
        )
        print(
            "TP topology      : "
            "8 GPUs within each node"
        )
        print(
            "DP topology      : "
            "2 replicas across nodes"
        )
        print(
            "DP gradient sync : "
            "Synchronous AllReduce"
        )
        print(
            "=" * 60
        )

    # Two DP replicas process two independent
    # micro-batches each step.
    tokens_per_step = (
        DP_SIZE
        * MICRO_BATCH_SIZE
        * SEQ_LEN
    )

    measured_times = []

    total_steps = (
        WARMUP_STEPS
        + TRAIN_STEPS
    )

    for step in range(total_steps):
        optimizer.zero_grad(
            set_to_none=True
        )

        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()

        # ----------------------------
        # Forward
        # ----------------------------
        logits = model(
            input_ids
        )

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                VOCAB_SIZE,
            ),
            labels.reshape(-1),
        )

        # ----------------------------
        # Backward
        # ----------------------------
        loss.backward()

        # ----------------------------
        # DP synchronization
        #
        # TP collectives occurred inside
        # forward/backward.
        #
        # Now synchronize corresponding
        # TP shards across DP replicas.
        # ----------------------------
        sync_dp_gradients(
            model,
            dp_group,
        )

        # ----------------------------
        # Optimizer
        # ----------------------------
        optimizer.step()

        torch.cuda.synchronize()

        local_step_time = (
            time.perf_counter()
            - start
        )

        # Whole hybrid job progresses at the
        # speed of the slowest of all 16 ranks.
        step_time = reduce_world_max(
            local_step_time,
            device,
        )

        local_peak_mem = (
            torch.cuda.max_memory_allocated(
                device
            )
        )

        peak_mem = reduce_world_max(
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

    peak_mem = reduce_world_max(
        local_peak_mem,
        device,
    )

    if rank == 0:
        print()
        print(
            "=" * 60
        )
        print(
            "TP + DP GPT 7B Result"
        )
        print(
            "=" * 60
        )
        print(
            f"TP size           : "
            f"{TP_SIZE}"
        )
        print(
            f"DP size           : "
            f"{DP_SIZE}"
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
