# Adapted from mindspeed_llm/tasks/high_availability/initialize_patch.py at effa8e78.
"""Bind the device and start TFT before creating the distributed world."""

import torch.distributed as dist


def initialize_distributed(args):
    """Perform the official initialize_distributed_wrapper prelude."""
    import torch_npu

    from .communication_patch import install_new_group_wrapper
    from .tft_train_initialize import tft_init_controller_processor

    if dist.is_initialized():
        raise RuntimeError("TFT must start before the job creates its distributed world")

    torch_npu.npu.set_device(args.local_rank)
    tft_init_controller_processor()
    if args.enable_worker_reboot:
        install_new_group_wrapper()
