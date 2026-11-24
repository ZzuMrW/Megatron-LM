# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Modification description：Modify the compose of processor group DP-CP-EP for MindIo.

from logging import getLogger

import torch

from megatron.core import mpu

ttp_logger = getLogger(__name__)

REPAIR_GROUP = None
REPAIR_GROUP_GLOO = None


def build_repair_group(rank_list: list):
    global REPAIR_GROUP, REPAIR_GROUP_GLOO
    REPAIR_GROUP = torch.distributed.new_group(rank_list, use_local_synchronization=True)
    REPAIR_GROUP_GLOO = torch.distributed.new_group(rank_list, use_local_synchronization=True, backend="gloo")
    ttp_logger.info(
        "[repair] rank:%s build repair group, rank list:%s, repair group:%s ",
        torch.distributed.get_rank(),
        rank_list,
        REPAIR_GROUP,
    )


def get_repair_group(use_gloo: bool = False):
    return REPAIR_GROUP_GLOO if use_gloo else REPAIR_GROUP


def destroy_repair_group():
    global REPAIR_GROUP, REPAIR_GROUP_GLOO
    if REPAIR_GROUP:
        torch.distributed.destroy_process_group(REPAIR_GROUP)
    REPAIR_GROUP = None
    if REPAIR_GROUP_GLOO:
        torch.distributed.destroy_process_group(REPAIR_GROUP_GLOO)
    REPAIR_GROUP_GLOO = None


def optimizer_domains(optimizer, args):
    """Map official ATTENTION/MOE indices to the actual native optimizer leaves."""
    dense = mpu.get_data_parallel_group(with_context_parallel=True)
    dense_intra = mpu.get_data_parallel_group(with_context_parallel=True, partial_data_parallel=True)
    configs = [(dense, dense_intra)]
    if args.num_experts is not None:
        configs.append(
            (
                mpu.get_expert_data_parallel_group(),
                mpu.get_expert_data_parallel_group(partial_expert_data_parallel=True),
            )
        )
    inter = tuple(torch.distributed.get_process_group_ranks(mpu.get_inter_distributed_optimizer_instance_group()))
    leaves = getattr(optimizer, 'chained_optimizers', [optimizer])
    result = []
    rank = torch.distributed.get_rank()
    instances = args.num_distributed_optimizer_instances
    for full, intra in configs:
        members = tuple(torch.distributed.get_process_group_ranks(full))
        shard_members = tuple(torch.distributed.get_process_group_ranks(intra))
        width = len(shard_members)
        instance, shard = divmod(members.index(rank), width)
        if (
            members != tuple(sorted(members))
            or len(members) != width * instances
            or members[instance * width : (instance + 1) * width] != shard_members
            or members[shard::width] != inter
        ):
            raise RuntimeError("TFT sorted replica table does not match native DistOpt shard order")
        bound = [(i, leaf) for i, leaf in enumerate(leaves) if getattr(leaf, 'data_parallel_group', None) is intra]
        result.append((members, width, bound))
    if sorted(index for _, _, domain_leaves in result for index, _ in domain_leaves) != list(range(len(leaves))):
        raise RuntimeError("TFT could not bind every native optimizer leaf to a unique domain")
    return result
