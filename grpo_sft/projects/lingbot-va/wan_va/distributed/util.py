# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized():
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    # if world_size > 1:
    torch.cuda.set_device(local_rank)
    # Default PG timeout for collectives (FSDP). NCCL watchdog also honors
    # TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC in the launch environment.
    timeout_sec = int(os.environ.get("LINGBOT_DIST_TIMEOUT_SEC", "7200"))
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_sec),
    )

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
