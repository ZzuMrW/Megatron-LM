# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
"""Port official dump selection, writer and rename to native 0.18.0 legacy files."""

import os
from argparse import Namespace
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
import torch.distributed as dist

from megatron.core import mpu
from megatron.core.utils import unwrap_model
from megatron.training import get_args
from megatron.training.checkpointing import (
    generate_state_dict,
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    get_distributed_optimizer_checkpoint_name,
)

from .native_distopt import ADAM_KEYS


def _save_root(args):
    # Keep the legacy dump separate from a periodic torch_dist tracker.
    return Path(args.save) if args.ckpt_format == 'torch' else Path(str(args.save) + '.tft')


def _temporary_directory(step):
    return _save_root(get_args()) / f'iter_{step:07d}_tmp'


def tft_save_callback(step, save_info, training, ctx):
    """Write the local legacy checkpoint contribution from TFT-selected shard ranks."""
    training.require_state(step)
    rank = dist.get_rank()
    selected = {}
    for entry in save_info:
        domain, ranks = entry['type'], tuple(entry['ranks'])
        if not 0 <= domain < len(training.domains) or domain in selected:
            raise ValueError("TFT save assignment has an invalid or duplicate optimizer domain")
        full, width, _ = training.domains[domain]
        if (
            rank not in ranks
            or any(member not in full for member in ranks)
            or sorted(full.index(member) % width for member in ranks) != list(range(width))
        ):
            raise ValueError("TFT save assignment must cover each logical shard exactly once")
        if full.index(ranks[0]) % width:
            raise ValueError("TFT save writer is outside the first logical shard")
        selected[domain] = ranks

    if not selected:
        raise ValueError("TFT did not assign a local save contribution")
    # A legacy model_optim_rng file needs all domains for its model coordinate.
    # Frozen TFT only supplies local assignments, so require a common selection.
    ranks = next(iter(selected.values()))
    if (
        set(selected) != set(range(len(training.domains)))
        or any(selection != ranks for selection in selected.values())
        or get_args().expert_model_parallel_size != 1
    ):
        raise ValueError("Legacy TFT dump needs a common native-domain writer for each model file")

    writer = ranks[0]
    group = dist.new_group(list(ranks), backend='gloo', use_local_synchronization=True)
    try:
        actual = dist.get_process_group_ranks(group)
        model = unwrap_model(training.model)
        weights, packed = {}, {}
        for domain in sorted(selected):
            members, width, leaves = training.domains[domain]
            order = sorted(range(len(actual)), key=lambda index: members.index(actual[index]) % width)
            shard = members.index(rank) % width
            for index, leaf in leaves:
                buffers, parameters = _gather_leaf(leaf, shard, width, order, writer, group)
                if rank == writer:
                    packed[index] = buffers
                    if weights.keys() & parameters.keys():
                        raise ValueError("Native optimizer leaves supplied overlapping parameters")
                    weights.update(parameters)
        if rank == writer:
            save_checkpoint(step, training, model, packed, weights)
    finally:
        dist.destroy_process_group(group)


def _gather_leaf(leaf, shard, width, order, writer, group):
    """Gather and reorder native coalesced legacy buffers, using the TFT global writer."""
    is_writer = dist.get_rank() == writer
    packed = {'buckets_coalesced': True}
    parameters = {}
    for buffer_index, by_dtype in enumerate(leaf.gbuf_ranges):
        buffer = leaf.buffers[buffer_index]
        for dtype, ranges in by_dtype.items():
            total = buffer.numel_unpadded
            state = {key: torch.zeros(total, dtype=torch.float32) for key in ADAM_KEYS} if is_writer else None
            offset = 0
            for bucket_index, ranges_by_parameter in enumerate(ranges):
                bucket = buffer.buckets[bucket_index]
                bucket_size = bucket.grad_data.numel()
                if bucket_size % width:
                    raise ValueError("Native bucket cannot be partitioned by the selected shards")
                size = bucket_size // width
                valid = bucket.numel_unpadded
                by_key = {key: torch.zeros(size, dtype=torch.float32) for key in ADAM_KEYS}
                bucket_parameters = {
                    param: (start - bucket.offset, end - bucket.offset)
                    for param, (start, end, owner) in buffer.param_index_map.items()
                    if owner == bucket_index
                }
                shard_start, shard_end = shard * size, (shard + 1) * size
                expected = {
                    param: (max(start, shard_start), min(end, shard_end))
                    for param, (start, end) in bucket_parameters.items()
                    if max(start, shard_start) < min(end, shard_end)
                }
                if expected.keys() != ranges_by_parameter['param_map'].keys():
                    raise ValueError("TFT dump shard coverage differs from the native bucket")
                for parameter, location in ranges_by_parameter['param_map'].items():
                    world = location['gbuf_world_in_bucket']
                    local = location['gbuf_local']
                    if not (
                        0 <= local.start < local.end <= size
                        and (world.start, world.end) == expected[parameter]
                        and (world.start, world.end) == (shard_start + local.start, shard_start + local.end)
                        and world.end <= valid
                    ):
                        raise ValueError("Native dump encountered a shard-range mismatch")
                    tensors = leaf._get_main_param_and_optimizer_states(parameter)
                    for key, piece in by_key.items():
                        value = tensors[key]
                        if value.dtype != torch.float32 or value.numel() != local.end - local.start:
                            raise ValueError("Native dump encountered incompatible Adam state")
                        piece[local.start : local.end].copy_(value.detach().view(-1).cpu())
                for key, piece in by_key.items():
                    received = [torch.empty_like(piece) for _ in order] if is_writer else None
                    dist.gather(piece, gather_list=received, dst=writer, group=group)
                    if is_writer:
                        state[key][offset : offset + valid].copy_(
                            torch.cat([received[index] for index in order])[:valid]
                        )
                if is_writer:
                    for parameter, (start, end) in bucket_parameters.items():
                        if not 0 <= start < end <= valid:
                            raise ValueError("Native dump encountered a parameter outside its bucket")
                        parameters[parameter] = (
                            state['param'][offset + start : offset + end].reshape(parameter.shape).to(parameter.dtype)
                        )
                offset += valid
            if offset != total:
                raise ValueError("Native dump coalesced layout differs from the reader")
            if is_writer:
                state['numel_unpadded'] = total
                packed.setdefault(buffer_index, {})[dtype] = state
    return packed, parameters


def save_checkpoint(step, training, model, packed, weights):
    """Emit the native legacy reader's model metadata and chained optimizer file."""
    args = Namespace(**vars(get_args()))
    args.ckpt_format, args.use_dist_ckpt = 'torch', False
    state = generate_state_dict(
        args,
        model,
        training.optimizer,
        training.scheduler,
        [training.progress['rng']],
        iteration=step,
    )
    state['num_floating_point_operations_so_far'] = training.progress['flops']
    for (index, name, attribute), buffer in training.progress['model_buffers'].items():
        model_state = state['model' if len(model) == 1 else f'model{index}']
        if not attribute and name in model_state:
            model_state[name] = buffer
    for parameter, aliases in training.names.items():
        if parameter.requires_grad and parameter not in weights:
            raise ValueError("TFT selected shards do not cover a trainable parameter")
        if parameter not in weights:
            continue
        for index, name in aliases:
            model_state = state['model' if len(model) == 1 else f'model{index}']
            if name not in model_state:
                raise ValueError(f"Native checkpoint lacks parameter {name}")
            model_state[name] = weights[parameter]
    name = Path(get_checkpoint_name(str(_save_root(args)), step)).parent.name
    checkpoint = _temporary_directory(step) / name / 'model_optim_rng.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if sorted(packed) != list(range(len(packed))):
        raise ValueError("TFT could not compose all native optimizer leaves")
    optimizer = packed[0] if len(packed) == 1 else [packed[i] for i in range(len(packed))]
    with open(get_distributed_optimizer_checkpoint_name(str(checkpoint)), 'xb') as stream:
        torch.save(optimizer, stream)
    with checkpoint.open('xb') as stream:
        torch.save(state, stream)


def tft_rename_callback(step, training):
    """Publish the completed temporary directory, then replace its legacy tracker."""
    root = _save_root(get_args())
    stage = _temporary_directory(step)
    destination = root / f'iter_{step:07d}'
    tracker = Path(get_checkpoint_tracker_filename(str(root)))
    if tracker.exists() and int(tracker.read_text()) >= step:
        raise ValueError("TFT will not replace a tracker at the same or later step")
    tp = mpu.get_tensor_model_parallel_world_size()
    pp = mpu.get_pipeline_model_parallel_world_size()
    for tensor_rank in range(tp):
        for pipeline_rank in range(pp):
            filename = get_checkpoint_name(
                str(root),
                step,
                tensor_rank=tensor_rank,
                pipeline_rank=pipeline_rank,
                expert_parallel=False,
            )
            model_file = stage / Path(filename).parent.name / 'model_optim_rng.pt'
            optimizer_file = Path(get_distributed_optimizer_checkpoint_name(str(model_file)))
            if any(not path.is_file() or path.stat().st_size == 0 for path in (model_file, optimizer_file)):
                raise ValueError(f"TFT dump lacks complete files for TP {tensor_rank} PP {pipeline_rank}")
    if destination.exists():
        raise FileExistsError(f"TFT checkpoint directory already exists: {destination}")
    stage.rename(destination)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(str(step))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, tracker)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
