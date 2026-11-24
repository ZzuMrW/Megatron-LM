# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""Port the official rollback order to native DDP and its iterator builder."""

from logging import getLogger

import torch
import torch.distributed as dist

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.core.rerun_state_machine import RerunState, get_rerun_state_machine
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.training import get_args, get_timers

from .dataloader_patch import build_dataset
from .tft_replica_group import destroy_repair_group
from .tft_tensor_recovery import restore_model_tensors

ttp_logger = getLogger(__name__)


@torch.no_grad()
def rollback_callback(step, training, ctx):
    from .tft_train_initialize import restore_rng

    args = get_args()
    if training.loaded:
        # The controller learns the actual disk step after the rollback callback.
        step = training.received['step']
    training.require_state(step)
    progress = training.received or training.original_rank_progress(step)
    training.damaged = True
    for key in training.counters:
        setattr(args, key, progress['counters'][key])
    args.iteration = args.curr_iteration = step
    args.num_floating_point_operations_so_far = progress['flops']
    training.scheduler.num_steps = 0
    training.scheduler.load_state_dict(progress['scheduler'])
    update_num_microbatches(args.consumed_train_samples, consistency_check=True)

    feature_rollback()
    restore_model_tensors(training.model, progress['model_buffers'])
    gather_model_params_from_optimizer(training, step)
    build_dataset(training)
    restore_rng(progress['rng'])
    dist.barrier()
    destroy_repair_group()
    training_log_repair(step, training)
    rebuild_global_vars(step, training)
    training.progress = progress
    training.received = None
    training.loaded = False
    training.damaged = False


def feature_rollback():
    mpu.destroy_global_memory_buffer()
    mpu._set_global_memory_buffer()
    destroy_moe_metrics_tracker()


def gather_model_params_from_optimizer(training, step):
    training.optimizer.zero_grad(set_to_none=True)
    for _, _, leaves in training.domains:
        for _, leaf in leaves:
            leaf._scale_one.fill_(1)
            leaf._copy_main_params_to_model_params()
    for model in training.model:
        if model.remove_forward_pre_hook_handles:
            model.disable_forward_pre_hook(param_sync=False)
        model.zero_grad_buffer()
        model.start_param_sync(force_sync=True)
    torch.npu.synchronize()
    ttp_logger.info("TFT restored native model shards at step %s", step)


def training_log_repair(step, training):
    if training.repaired_log_args is not None:
        ttp_logger.info("TFT restored training log values at step %s: %s", step, training.repaired_log_args)
        training.log_args = training.repaired_log_args
        training.repaired_log_args = None


def rebuild_global_vars(step, training):
    for key, value in training.sync_hooks.items():
        setattr(training.config, key, value)
    for timer in get_timers()._timers.values():
        timer.reset()
    rerun = get_rerun_state_machine()
    rerun.current_iteration = step
    rerun.state = RerunState.NOT_RUNNING_YET
