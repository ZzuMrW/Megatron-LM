# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""Port the official TFT UCE/ARF repair dispatch and peer transfer order."""

import io

import torch
import torch.distributed as dist
from mindio_ttp.framework_ttp import RepairType, tft_report_load_ckpt_step

from megatron.training import get_args
from megatron.training.checkpointing import load_checkpoint

from .native_distopt import apply_leaf, export_leaf, stage_leaf
from .tft_replica_group import build_repair_group, get_repair_group


def repair_callback(step, need_rebuild, error_ranks, repair_info, training, ctx):
    """Execute the TFT-assigned action. The SDK owns donor and fallback decisions."""
    kind = RepairType(repair_info['repair_type'])
    rank = dist.get_rank()
    participants = tuple(repair_info['rank_list'])
    records = sorted(
        zip(repair_info['type'], repair_info['src'], repair_info['dst'], strict=True),
        key=lambda record: (record[1], record[2], record[0]),
    )
    if kind == RepairType.RT_UCE_LOWLEVEL and all(src == dst == rank for _, src, dst in records):
        training.require_state(step)
        return
    if kind in (RepairType.RT_LOAD_CKPT, RepairType.RT_LOAD_REBUILD):
        if set(participants) != set(range(dist.get_world_size())):
            raise ValueError("The native checkpoint reader requires a whole-world TFT load action")
        load_ckpt_repair(training)
        return

    is_sender = kind == RepairType.RT_SEND
    if not is_sender and kind not in (
        RepairType.RT_UCE_HIGHLEVEL,
        RepairType.RT_UCE_LOWLEVEL,
        RepairType.RT_RECV_REPAIR,
    ):
        raise ValueError(f"Unsupported TFT repair action {kind}")
    for domain, source, destination in records:
        if domain < 0:
            raise ValueError("TFT peer assignment uses an unknown optimizer domain")
        members, width, _ = training.domains[domain]
        if (
            source == destination
            or rank != (source if is_sender else destination)
            or members.index(source) % width != members.index(destination) % width
        ):
            raise ValueError("TFT peer assignment does not match a native replica shard")

    build_repair_group(participants)
    if is_sender:
        send_rank_repair(step, records, training)
    else:
        recv_rank_repair(step, records, training)


def send_rank_repair(step, records, training):
    training.require_state(step)
    exports = []
    for domain, _, destination in records:
        leaves, tensors = {}, []
        for index, leaf in training.domains[domain][2]:
            state, owned = export_leaf(leaf, training.names)
            leaves[index] = state
            tensors.extend(owned)
        save_and_send_ckpt(destination, {'step': step, 'domain': domain, 'leaves': leaves})
        exports.append(tensors)

    for (_, _, destination), tensors in zip(records, exports, strict=True):
        for tensor in tensors:
            dist.send(tensor, dst=destination, group=get_repair_group())
    for _, _, destination in records:
        send_log_args(destination, training)


def recv_rank_repair(step, records, training):
    domains = {domain for domain, _, _ in records}
    require_all = training.step != step or training.pending is not None or training.damaged
    if require_all and domains != set(range(len(training.domains))):
        raise ValueError("TFT assignments do not restore all potentially invalid native leaves")
    progress = training.original_rank_progress(step)
    grouped = []
    for domain, source, _ in records:
        metadata = recv_ckpt_from_peer(source)
        if metadata['step'] != step or metadata['domain'] != domain:
            raise ValueError("TFT peer returned a different step or optimizer domain")
        stages = [
            (leaf, stage_leaf(leaf, training.names, metadata['leaves'][index]))
            for index, leaf in training.domains[domain][2]
        ]
        grouped.append((source, stages))

    updates = []
    for source, stages in grouped:
        for leaf, stage in stages:
            tensors = []
            for _, specs, _, _ in stage[1]:
                for _, shape, dtype in specs:
                    tensor = torch.empty(shape, dtype=dtype, device='npu')
                    dist.recv(tensor, src=source, group=get_repair_group())
                    tensors.append(tensor.cpu())
            updates.append((leaf, stage, tensors))
    for _, source, _ in records:
        recv_log_args(source, training)

    training.damaged = True
    for leaf, stage, tensors in updates:
        apply_leaf(leaf, stage, tensors)
    torch.npu.synchronize()
    training.received = progress
    training.step, training.pending, training.damaged = step, None, False


def save_and_send_ckpt(destination, state_dict):
    """Send the same size-then-serialized-NPU-tensor metadata frames as 26.2.0."""
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    payload = torch.frombuffer(bytearray(buffer.getbuffer()), dtype=torch.uint8).to('npu')
    size = torch.tensor([payload.numel()], dtype=torch.long, device='npu')
    dist.send(size, dst=destination, group=get_repair_group())
    dist.send(payload, dst=destination, group=get_repair_group())


def recv_ckpt_from_peer(source):
    size = torch.empty(1, dtype=torch.long, device='npu')
    dist.recv(size, src=source, group=get_repair_group())
    payload = torch.empty(size.item(), dtype=torch.uint8, device='npu')
    dist.recv(payload, src=source, group=get_repair_group())
    buffer = io.BytesIO(payload.cpu().numpy().tobytes())
    return torch.load(buffer, map_location='cpu', weights_only=False)


def send_log_args(destination, training):
    """Keep the official LM loss, grad norm and zero-count frame order."""
    loss, norm, zeros = training.log_args
    size = torch.tensor([-1 if loss is None else 1], dtype=torch.long)
    dist.send(size, dst=destination, group=get_repair_group(True))
    if loss is not None:
        dist.send(
            torch.as_tensor(loss, dtype=torch.float32, device='npu').reshape(1),
            dst=destination,
            group=get_repair_group(),
        )
    for value in (norm, zeros):
        tensor = torch.tensor([-1 if value is None else value], dtype=torch.float32)
        dist.send(tensor, dst=destination, group=get_repair_group(True))


def recv_log_args(source, training):
    size = torch.empty(1, dtype=torch.long)
    dist.recv(size, src=source, group=get_repair_group(True))
    loss = None
    if size.item() == 1:
        loss = torch.empty(1, dtype=torch.float32, device='npu')
        dist.recv(loss, src=source, group=get_repair_group())
        loss = loss.item()
    stats = []
    for _ in range(2):
        value = torch.empty(1, dtype=torch.float32)
        dist.recv(value, src=source, group=get_repair_group(True))
        stats.append(None if value.item() == -1 else value.item())
    training.repaired_log_args = (loss, *stats)


def load_ckpt_repair(training):
    """Load the native checkpoint only for TFT's two disk actions."""
    args = get_args()
    if args.no_load_optim or args.no_load_rng:
        raise ValueError("TFT disk recovery requires optimizer and RNG state")
    training.damaged = True
    for field in training.counters:
        setattr(args, field, 0)
    training.scheduler.num_steps = 0
    step, flops = load_checkpoint(
        training.model, training.optimizer, training.scheduler, checkpointing_context=training.checkpointing
    )
    if step <= 0:
        raise ValueError("TFT requires a positive checkpoint step")
    torch.npu.synchronize()
    training.received = training._progress(step, flops)
    training.step, training.pending, training.damaged = step, None, False
    training.loaded = True
    tft_report_load_ckpt_step(step)
