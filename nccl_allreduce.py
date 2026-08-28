```python
import os
import time
import torch
import torch.distributed as dist


# ============================================================
# Configuration
# ============================================================

MIN_SIZE = 8 * 1024                    # 8 KiB
MAX_SIZE = 8 * 1024 * 1024 * 1024      # 8 GiB

WARMUP_ITERS = 10


def get_iterations(size_bytes):
    """
    Use fewer iterations for large messages so that the
    benchmark does not take unnecessarily long.
    """

    if size_bytes <= 8 * 1024 * 1024:
        return 100          # <= 8 MiB

    elif size_bytes <= 64 * 1024 * 1024:
        return 50           # 16-64 MiB

    elif size_bytes <= 512 * 1024 * 1024:
        return 20           # 128-512 MiB

    elif size_bytes <= 2 * 1024 * 1024 * 1024:
        return 10           # 1-2 GiB

    else:
        return 5            # 4-8 GiB


def format_size(size_bytes):
    """
    Convert bytes into human-readable binary units.
    """

    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024 ** 3):.0f} GiB"

    elif size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024 ** 2):.0f} MiB"

    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KiB"

    else:
        return f"{size_bytes} B"


def main():

    # ========================================================
    # Distributed initialization
    # ========================================================

    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    # ========================================================
    # Print topology information
    # ========================================================

    if rank == 0:
        print()
        print("============================================================")
        print("PyTorch NCCL AllReduce Benchmark")
        print("============================================================")
        print(f"World size       : {world_size}")
        print(f"GPUs per node    : 8")
        print(f"Nodes            : 2")
        print(f"GPU backend      : NCCL")
        print(f"CUDA device      : {device}")
        print()
        print(
            f"{'Size':>10} "
            f"{'Elements':>14} "
            f"{'Iters':>6} "
            f"{'Time(ms)':>12} "
            f"{'AlgBW(GB/s)':>14} "
            f"{'BusBW(GB/s)':>14} "
            f"{'BusBW(Gb/s)':>14}"
        )
        print("-" * 100)

    # ========================================================
    # Test message sizes
    #
    # 8 KiB -> 16 KiB -> ... -> 8 GiB
    # ========================================================

    size_bytes = MIN_SIZE

    while size_bytes <= MAX_SIZE:

        # ----------------------------------------------------
        # Tensor size
        #
        # float32 = 4 bytes
        # ----------------------------------------------------

        num_elements = size_bytes // 4

        tensor = torch.ones(
            num_elements,
            dtype=torch.float32,
            device=device
        )

        # ----------------------------------------------------
        # Warmup
        # ----------------------------------------------------

        for _ in range(WARMUP_ITERS):
            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM
            )

        # Make sure all CUDA operations have completed
        torch.cuda.synchronize()

        # Synchronize all ranks before benchmark
        dist.barrier()

        # ----------------------------------------------------
        # Benchmark iterations
        # ----------------------------------------------------

        iterations = get_iterations(size_bytes)

        start = time.perf_counter()

        for _ in range(iterations):
            dist.all_reduce(
                tensor,
                op=dist.ReduceOp.SUM
            )

        # Wait until all NCCL CUDA operations finish
        torch.cuda.synchronize()

        end = time.perf_counter()

        # ----------------------------------------------------
        # Calculate average latency
        # ----------------------------------------------------

        avg_time_s = (end - start) / iterations
        avg_time_ms = avg_time_s * 1000

        # ----------------------------------------------------
        # Algorithm bandwidth
        #
        # algbw = message size / time
        #
        # This is the logical bandwidth from the collective
        # algorithm perspective.
        # ----------------------------------------------------

        algbw_GBps = (
            size_bytes
            / avg_time_s
            / 1e9
        )

        # ----------------------------------------------------
        # NCCL-style bus bandwidth
        #
        # AllReduce:
        #
        # busbw =
        #     size * 2*(N-1)/N / time
        #
        # For N = 16:
        #
        # coefficient = 2 * 15 / 16
        #              = 1.875
        #
        # This estimates the actual communication traffic
        # carried by the network.
        # ----------------------------------------------------

        busbw_GBps = (
            size_bytes
            * 2
            * (world_size - 1)
            / world_size
            / avg_time_s
            / 1e9
        )

        busbw_Gbps = busbw_GBps * 8

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        if rank == 0:

            print(
                f"{format_size(size_bytes):>10} "
                f"{num_elements:>14} "
                f"{iterations:>6} "
                f"{avg_time_ms:>12.3f} "
                f"{algbw_GBps:>14.2f} "
                f"{busbw_GBps:>14.2f} "
                f"{busbw_Gbps:>14.2f}"
            )

        # ----------------------------------------------------
        # Release tensor before next message size
        # ----------------------------------------------------

        del tensor

        torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Next message size = 2x
        # ----------------------------------------------------

        size_bytes *= 2

    # ========================================================
    # Finish
    # ========================================================

    if rank == 0:
        print("-" * 100)
        print()
        print("Benchmark completed.")
        print("============================================================")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```
