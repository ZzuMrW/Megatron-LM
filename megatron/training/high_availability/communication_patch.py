# Adapted from mindspeed_llm/tasks/high_availability/communication_patch.py at effa8e78.
"""Keep replacement startup local and retain native group creation arguments."""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import torch.distributed as dist

_original_new_group = None
_original_barrier_setting = None
inventory = []


@dataclass(eq=False)
class DeferredGloo:
    ranks: tuple
    local_rank: int

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(self.local_rank)


@dataclass
class Group:
    ranks: tuple
    backend: str
    purpose: str
    kwargs: dict
    handle: object


def install_new_group_wrapper():
    """Record ordered membership, including Gloo groups deferred on the replacement."""
    from .tft_arf_group_repair import tft_is_arf_reboot_node

    global _original_new_group, _original_barrier_setting
    _original_new_group = dist.new_group
    _original_barrier_setting = os.environ.get('TORCH_DIST_INIT_BARRIER')
    if tft_is_arf_reboot_node():
        os.environ['TORCH_DIST_INIT_BARRIER'] = '0'
    dist.new_group = new_group_wrapper


def new_group_wrapper(
    ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False, group_desc=None
):
    from .tft_arf_group_repair import tft_is_arf_reboot_node

    members = tuple(range(dist.get_world_size()) if ranks is None else sorted(ranks))
    actual_backend = backend or dist.get_backend()
    if actual_backend not in ('hccl', 'gloo'):
        raise ValueError(f"ARF cannot reconnect native backend {actual_backend}")
    kwargs = dict(
        ranks=list(members),
        timeout=timeout,
        backend=backend,
        pg_options=pg_options,
        use_local_synchronization=actual_backend == 'hccl',
        group_desc=group_desc,
    )
    if tft_is_arf_reboot_node() and actual_backend == 'gloo':
        rank = dist.get_rank()
        handle = DeferredGloo(members, rank) if rank in members else dist.GroupMember.NON_GROUP_MEMBER
    else:
        handle = _original_new_group(**kwargs)
    inventory.append(Group(members, actual_backend, group_desc or '', kwargs, handle))
    return handle


def seal_group_inventory():
    """Stop capturing group creation before TFT starts invoking repair callbacks."""
    if _original_new_group is None:
        return
    dist.new_group = _original_new_group
    if _original_barrier_setting is None:
        os.environ.pop('TORCH_DIST_INIT_BARRIER', None)
    else:
        os.environ['TORCH_DIST_INIT_BARRIER'] = _original_barrier_setting


@contextmanager
def replacement_model_setup(recovery):
    """Permit only startup actions with a known original-rank result."""
    from megatron.core import mpu

    original_gather = dist.all_gather_object
    original_reduce = dist.all_reduce
    exchange = 0

    def gather(output, value, group=None):
        nonlocal exchange
        caller = sys._getframe(1).f_globals.get('__name__')
        if caller != 'megatron.core.optimizer' or group is not None or recovery is None:
            raise ValueError("ARF needs the original optimizer parameter-group key exchange")
        rank = dist.get_rank()
        rows = recovery.optimizer_group_keys(rank, exchange)
        if rows is None or len(rows) != len(output) or rows[rank] != value:
            raise ValueError("ARF optimizer parameter-group keys differ from original startup")
        output[:] = rows
        exchange += 1

    def reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        caller = sys._getframe(1).f_globals.get('__name__')
        if (
            caller != 'megatron.core.models.common.language_module.language_module'
            or group is None
            or group is not mpu.get_embedding_group(check_initialized=False)
            or op != dist.ReduceOp.SUM
            or async_op
        ):
            raise ValueError(f"ARF cannot defer startup reduction from {caller}")
        # Both tied-weight copies are restored from native DistOpt before training.

    def reject(name):
        def unexpected(*args, **kwargs):
            raise ValueError(f"ARF startup collective {name} needs a customer-provided result")

        return unexpected

    guarded = {}
    dist.all_gather_object, dist.all_reduce = gather, reduce
    try:
        for name in (
            'barrier',
            'broadcast',
            'broadcast_object_list',
            'all_gather',
            'all_gather_into_tensor',
            'reduce_scatter_tensor',
            'all_to_all_single',
        ):
            guarded[name] = getattr(dist, name)
            setattr(dist, name, reject(name))
        yield
    finally:
        dist.all_gather_object, dist.all_reduce = original_gather, original_reduce
        for name, function in guarded.items():
            setattr(dist, name, function)
