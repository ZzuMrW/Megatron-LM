# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Port the official ARF group-rebuild callback to native 0.18.0 group ownership."""

import os
from logging import getLogger

import torch.distributed as dist

from megatron.core import mpu
from megatron.training import get_args, get_timers

from .communication_patch import DeferredGloo, inventory

ttp_logger = getLogger(__name__)
ARF_REBOOT_FLAG = False
LOAD = None
PRETRAINED_CHECKPOINT = None


def tft_is_arf_reboot_node():
    return ARF_REBOOT_FLAG


def update_arf_reboot_flag(new_state):
    global ARF_REBOOT_FLAG, LOAD, PRETRAINED_CHECKPOINT
    args = get_args()
    if new_state:
        LOAD = args.load
        PRETRAINED_CHECKPOINT = args.pretrained_checkpoint
        args.load = None
        args.pretrained_checkpoint = None
    elif ARF_REBOOT_FLAG:
        args.load = LOAD
        args.pretrained_checkpoint = PRETRAINED_CHECKPOINT
    ARF_REBOOT_FLAG = new_state


def _phase(record):
    purpose = record.purpose
    if purpose.startswith('DATA_PARALLEL_GROUP_WITH_CP'):
        return 1
    if purpose.startswith('DATA_PARALLEL_GROUP'):
        return 0
    if purpose.startswith('CONTEXT_PARALLEL_GROUP'):
        return 2
    if purpose == 'MODEL_PARALLEL_GROUP':
        return 3
    if purpose == 'TENSOR_MODEL_PARALLEL_GROUP':
        return 4
    if purpose in ('PIPELINE_MODEL_PARALLEL_GROUP', 'EMBEDDING_GROUP', 'POSITION_EMBEDDING_GROUP'):
        return 5
    if (
        'PARTIAL' in purpose
        or 'OPTIMIZER_INSTANCE' in purpose
        or purpose.endswith('_L1')
        or purpose.startswith('HIERARCHICAL_EXPERT_DATA_PARALLEL_GROUP')
    ):
        return 7
    return 6


def arf_rebuild_process_group_callback(fault_ranks, training, ctx):
    """Reconnect world, parallel groups and optimizer groups in the TFT ARF callback."""
    if not get_args().enable_worker_reboot or not inventory:
        raise ValueError("TFT ARF has no registered Megatron group inventory")
    rank = dist.get_rank()
    if (
        not fault_ranks
        or len(set(fault_ranks)) != len(fault_ranks)
        or any(type(fault) is not int or not 0 <= fault < dist.get_world_size() for fault in fault_ranks)
        or (rank in fault_ranks) != tft_is_arf_reboot_node()
    ):
        raise ValueError("TFT ARF replacement list differs from local processor status")

    timers = get_timers()
    timers('interval-time', log_level=0).reset()
    previous = os.environ.get('TORCH_DIST_INIT_BARRIER')
    os.environ['TORCH_DIST_INIT_BARRIER'] = '1'
    try:
        _reconnect(None)
        replacements, old_groups = {}, []
        for record in sorted(inventory, key=_phase):
            if rank not in record.ranks:
                continue
            if record.backend == 'hccl':
                _reconnect(record.handle)
                if tuple(dist.get_process_group_ranks(record.handle)) != record.ranks:
                    raise RuntimeError("ARF changed native HCCL shard order")
                continue
            old = record.handle
            record.handle = dist.new_group(**dict(record.kwargs, use_local_synchronization=True))
            if tuple(dist.get_process_group_ranks(record.handle)) != record.ranks:
                raise RuntimeError("ARF changed native Gloo shard order")
            replacements[id(old)] = record.handle
            if not isinstance(old, DeferredGloo):
                old_groups.append(old)

        update_model_and_optim_related_group(training, replacements)
        for group in old_groups:
            dist.destroy_process_group(group)
        dist.barrier()
        update_arf_reboot_flag(False)
        training.rebuilt = True
    finally:
        if previous is None:
            os.environ.pop('TORCH_DIST_INIT_BARRIER', None)
        else:
            os.environ['TORCH_DIST_INIT_BARRIER'] = previous
    ttp_logger.info("TFT rank %d restored native process groups", rank)


def _reconnect(group):
    world = dist.group.WORLD
    expected = world if group is None else group
    result = dist.reinit_process_group(group=group, rebuild_link=True)
    if dist.group.WORLD is not world or (result is not None and result is not expected):
        raise RuntimeError("ARF requires TorchNPU in-place HCCL process-group reconnection")


def update_model_and_optim_related_group(training, replacements):
    """Replace Gloo references; native DDP and schedule HCCL handles survive in place."""
    for name, value in tuple(vars(mpu).items()):
        if id(value) in replacements:
            setattr(mpu, name, replacements[id(value)])
    mpu._global_process_group_list[:] = [
        replacements.get(id(group), group) for group in mpu._global_process_group_list
    ]
    for _, _, leaves in training.domains:
        for _, leaf in leaves:
            old = leaf.data_parallel_group_gloo
            leaf.data_parallel_group_gloo = replacements.get(id(old), old)
    training.checkpointing.clear()
