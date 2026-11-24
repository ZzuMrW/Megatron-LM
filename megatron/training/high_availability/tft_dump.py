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
    path = Path(args.save)
    return path if args.ckpt_format == 'torch' else Path(f'{path}.tft')


def _temporary_directory(step):
    return _save_root(get_args()) / f'iter_{step:07d}_tmp'


def tft_save_callback(step, save_info, training, ctx):
    """Write the local legacy checkpoint contribution from TFT-selected shard ranks."""
    training.require_state(step)
    rank = dist.get_rank()
    selected = {entry['type']: tuple(entry['ranks']) for entry in save_info}
    if set(selected) != set(range(len(training.domains))) or get_args().expert_model_parallel_size != 1:
        raise ValueError("Legacy TFT dump needs a common native-domain writer for each model file")

    ranks = selected[0]
    for domain, (members, width, _) in enumerate(training.domains):
        if selected[domain] != ranks:
            raise ValueError("Legacy TFT dump needs the same selected ranks for each optimizer domain")
        if rank not in ranks or sorted(members.index(member) % width for member in ranks) != list(range(width)):
            raise ValueError("TFT save assignment must cover each native shard exactly once")

    writer = ranks[0]
    group = dist.new_group(list(ranks), backend='gloo', use_local_synchronization=True)
    try:
        actual = dist.get_process_group_ranks(group)
        model = unwrap_model(training.model)
        weights, packed = {}, {}
        for members, width, leaves in training.domains:
            order = sorted(range(len(actual)), key=lambda index: members.index(actual[index]) % width)
            for index, leaf in leaves:
                buffers, parameters = _gather_leaf(leaf, order, writer, group)
                if rank == writer:
                    packed[index] = buffers
                    weights.update(parameters)
        if rank == writer:
            save_checkpoint(step, training, model, packed, weights)
    finally:
        dist.destroy_process_group(group)


def _gather_leaf(leaf, order, writer, group):
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
                size = bucket.grad_data.numel() // len(order)
                valid = bucket.numel_unpadded
                by_key = {key: torch.zeros(size, dtype=torch.float32) for key in ADAM_KEYS}
                for parameter, location in ranges_by_parameter['param_map'].items():
                    local = location['gbuf_local']
                    tensors = leaf._get_main_param_and_optimizer_states(parameter)
                    for key, piece in by_key.items():
                        piece[local.start : local.end].copy_(tensors[key].detach().view(-1).cpu())
                for key, piece in by_key.items():
                    received = [torch.empty_like(piece) for _ in order] if is_writer else None
                    dist.gather(piece, gather_list=received, dst=writer, group=group)
                    if is_writer:
                        state[key][offset : offset + valid].copy_(
                            torch.cat([received[index] for index in order])[:valid]
                        )
                if is_writer:
                    for parameter, (start, end, owner) in buffer.param_index_map.items():
                        if owner == bucket_index:
                            left, right = offset + start - bucket.offset, offset + end - bucket.offset
                            parameters[parameter] = (
                                state['param'][left:right].reshape(parameter.shape).to(parameter.dtype)
                            )
                offset += valid
            if is_writer:
                state['numel_unpadded'] = total
                packed.setdefault(buffer_index, {})[dtype] = state
    return packed, parameters


def save_checkpoint(step, training, model, packed, weights):
    """Emit the native legacy reader's model metadata and chained optimizer file."""
    args = Namespace(**vars(get_args()))
    if args.no_save_optim or args.no_save_rng:
        raise ValueError("TFT legacy checkpoint requires optimizer and RNG state")
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
    stage.rename(destination)
    stream = NamedTemporaryFile(mode='w', dir=root, delete=False)
    try:
        with stream:
            stream.write(str(step))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stream.name, tracker)
    finally:
        Path(stream.name).unlink(missing_ok=True)
