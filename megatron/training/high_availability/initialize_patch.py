# Adapted from mindspeed_llm/tasks/high_availability/initialize_patch.py at effa8e78.
"""TFT startup and checks for the native Partial DistOpt entry point."""

import os
from importlib.metadata import distribution
from pathlib import Path

import torch.distributed as dist


def initialize_distributed(args):
    """Perform the official initialize_distributed_wrapper prelude."""
    if dist.is_initialized():
        raise RuntimeError("TFT must start before the job creates its distributed world")
    if not args.use_distributed_optimizer or args.num_distributed_optimizer_instances < 2:
        raise ValueError("TFT requires native distributed optimizer instances")
    if not args.bf16 or args.optimizer != 'adam':
        raise ValueError("TFT port supports native BF16 Adam")
    if args.distributed_backend not in ('nccl', 'hccl') or args.dataloader_type not in ('single', 'cyclic'):
        raise ValueError("TFT requires Ascend HCCL and a native sampler")
    if args.skip_train or args.rerun_mode != 'disabled' or args.num_workers != 0:
        raise ValueError("TFT requires training, --rerun-mode disabled and --num-workers 0")
    if not args.save or not args.use_gloo_process_groups:
        raise ValueError("TFT requires --save and Megatron Gloo groups")
    excluded = (
        'use_megatron_fsdp',
        'use_torch_fsdp2',
        'use_custom_fsdp',
        'use_layer_wise_distributed_optimizer',
        'use_precision_aware_optimizer',
        'loss_scale',
        'optimizer_cpu_offload',
        'fp8',
        'fp4',
        'data_parallel_random_init',
        'perform_rl_step',
        'vision_pretraining',
        'lazy_mpu_init',
        'inprocess_restart',
        'tp_comm_overlap',
        'use_sharp',
        'sharp_enabled_group',
        'optimizer_cuda_graph',
        'full_validation',
        'multiple_validation_sets',
        'phase_transition_iterations',
        'fake_process_group',
        'async_save',
        'non_persistent_ckpt_type',
        'force_create_nccl_p2p_group',
        'no_load_optim',
        'no_load_rng',
        'no_save_optim',
        'no_save_rng',
        'finetune',
        'log_progress',
        'moe_use_upcycling',
        'ckpt_convert_format',
        'dump_param_to_param_group_map',
    )
    unsupported = [name for name in excluded if getattr(args, name, None)]
    if unsupported or args.cuda_graph_impl != 'none':
        raise ValueError(
            f"TFT native adapter configuration is unsupported: {unsupported}, graph={args.cuda_graph_impl}"
        )
    device = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if device != args.local_rank:
        raise ValueError("TFT LOCAL_RANK differs from Megatron local_rank")
    verify_tft_distribution()
    import torch_npu

    from .communication_patch import install_new_group_wrapper
    from .tft_train_initialize import tft_init_controller_processor

    torch_npu.npu.set_device(device)
    if (args.enable_hbmfault_repair or args.enable_worker_reboot) and not callable(
        getattr(dist, 'reinit_process_group', None)
    ):
        raise RuntimeError("TFT recovery requires TorchNPU reinit_process_group")
    tft_init_controller_processor()
    if args.enable_worker_reboot:
        install_new_group_wrapper()


def verify_tft_distribution():
    """Check the official wheel provenance before importing the native SDK."""
    wheel = distribution('mindio_ttp')
    version = Path(wheel.locate_file('mindio_ttp/VERSION')).read_text(encoding='utf-8').splitlines()
    # The release build records the latest commit affecting component/mindio/tft.
    if wheel.version != '26.2.0' or version != [
        'mindio_ttp version info:',
        'mindio_ttp version: 26.2.0',
        'git: 9a28823c002d113d109c369e4faf8790ddcd8d0a',
    ]:
        raise RuntimeError("TFT requires the official MindIO 26.2.0 component build")
