import os
import time
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================
# Model
# ============================================================

class GPTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )

        self.ln2 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)

        attn_out, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=True,
        )

        x = x + attn_out
        x = x + self.mlp(self.ln2(x))

        return x


class SmallGPT(nn.Module):
    def __init__(
        self,
        vocab_size=32000,
        seq_len=1024,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
    ):
        super().__init__()

        self.seq_len = seq_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.position_embedding = nn.Embedding(
            seq_len,
            hidden_size,
        )

        self.blocks = nn.ModuleList([
            GPTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
    
        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)
    
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )
    
        # Causal mask:
        # True = this position is NOT allowed to attend
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=input_ids.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
    
        for block in self.blocks:
            x = block(
                x,
                attn_mask=causal_mask,
            )
    
        x = self.ln_f(x)
        logits = self.lm_head(x)
    
        return logits


# ============================================================
# Distributed
# ============================================================

def setup():

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=device,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return rank, world_size, local_rank, device


def cleanup():
    dist.destroy_process_group()


# ============================================================
# Main
# ============================================================

def main():

    rank, world_size, local_rank, device = setup()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    torch.manual_seed(1234 + rank)
    torch.cuda.manual_seed(1234 + rank)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    vocab_size = 32000
    seq_len = 1024

    hidden_size = 1024
    num_layers = 12
    num_heads = 16

    batch_size = 2

    warmup_steps = 5
    train_steps = 20

    learning_rate = 3e-4

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SmallGPT(
        vocab_size=vocab_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
    ).to(device=device, dtype=torch.bfloat16)

    num_params = sum(
        p.numel()
        for p in model.parameters()
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        bucket_cap_mb=25,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    loss_fn = nn.CrossEntropyLoss()

    if rank == 0:
        print()
        print("=" * 60)
        print("DDP GPT Stage 1")
        print("=" * 60)
        print(f"World size       : {world_size}")
        print(f"Parameters       : {num_params / 1e6:.2f} M")
        print(f"Hidden size      : {hidden_size}")
        print(f"Layers           : {num_layers}")
        print(f"Heads            : {num_heads}")
        print(f"Sequence length  : {seq_len}")
        print(f"Batch/GPU        : {batch_size}")
        print(f"Global batch     : {batch_size * world_size}")
        print(f"Precision        : BF16")
        print("=" * 60)
        print()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    step_times = []

    total_steps = warmup_steps + train_steps

    for step in range(total_steps):

        # Each rank gets different random tokens
        input_ids = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device=device,
        )

        labels = torch.randint(
            low=0,
            high=vocab_size,
            size=(batch_size, seq_len),
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        dist.barrier()
        torch.cuda.synchronize()

        start = time.perf_counter()

        logits = model(input_ids)

        loss = loss_fn(
            logits.view(-1, vocab_size),
            labels.view(-1),
        )

        loss.backward()

        optimizer.step()

        torch.cuda.synchronize()

        end = time.perf_counter()

        step_time = end - start

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        if step >= warmup_steps:
            step_times.append(step_time)

        # Average loss across all ranks for display only
        loss_tensor = loss.detach().float()

        dist.all_reduce(
            loss_tensor,
            op=dist.ReduceOp.SUM,
        )

        avg_loss = loss_tensor.item() / world_size

        if rank == 0:

            tokens_per_step = (
                batch_size
                * seq_len
                * world_size
            )

            tokens_per_sec = (
                tokens_per_step
                / step_time
            )

            allocated = (
                torch.cuda.max_memory_allocated(device)
                / 1024**3
            )

            label = (
                "WARMUP"
                if step < warmup_steps
                else "TRAIN "
            )

            print(
                f"{label} "
                f"step={step:02d} "
                f"loss={avg_loss:.4f} "
                f"time={step_time*1000:.2f} ms "
                f"tokens/s={tokens_per_sec:,.0f} "
                f"max_mem={allocated:.2f} GiB",
                flush=True,
            )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    avg_step_time = sum(step_times) / len(step_times)

    global_tokens_per_step = (
        batch_size
        * seq_len
        * world_size
    )

    throughput = (
        global_tokens_per_step
        / avg_step_time
    )

    if rank == 0:
        print()
        print("=" * 60)
        print("DDP GPT Stage 1 Result")
        print("=" * 60)
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
            f"{global_tokens_per_step:,}"
        )
        print("=" * 60)

    cleanup()


if __name__ == "__main__":
    main()
