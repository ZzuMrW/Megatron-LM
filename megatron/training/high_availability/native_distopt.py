"""Native 0.18.0 shard access for the ported TFT callbacks."""

import math

import torch

from megatron.core.optimizer.optimizer import param_group_identifier_keys

ADAM_KEYS = ('param', 'exp_avg', 'exp_avg_sq')


def owned_ranges(leaf, names):
    for buffer, by_dtype in enumerate(leaf.gbuf_ranges):
        for dtype, buckets in by_dtype.items():
            for bucket, ranges in enumerate(buckets):
                for param, intervals in ranges['param_map'].items():
                    world = intervals['gbuf_world_in_bucket']
                    local = intervals['param']
                    yield param, (names[param], buffer, bucket, dtype, world.start, world.end, local.start, local.end)


def _master(leaf, parameter):
    group, index = leaf.model_param_group_index_map[parameter]
    return group, leaf.optimizer.param_groups[group]['params'][index]


def export_leaf(leaf, names):
    """Separate native metadata and tensors for the official two-phase peer transfer."""
    entries, tensors = [], []
    groups = [{key: value for key, value in group.items() if key != 'params'} for group in leaf.optimizer.param_groups]
    for parameter, extent in owned_ranges(leaf, names):
        _, main = _master(leaf, parameter)
        native = leaf.optimizer.state.get(main, {})
        values = leaf._get_main_param_and_optimizer_states(parameter)
        if set(native) - {'exp_avg', 'exp_avg_sq', 'step'}:
            raise ValueError("TFT supports native Adam state keys")
        keys = tuple(key for key in ADAM_KEYS if key in values)
        if keys not in (('param',), ADAM_KEYS):
            raise ValueError("TFT found incomplete Adam moments")
        entries.append(
            (extent, tuple((key, values[key].shape, values[key].dtype) for key in keys), native.get('step'))
        )
        tensors.extend(values[key].detach().contiguous() for key in keys)
    return {'groups': groups, 'entries': entries}, tensors


def stage_leaf(leaf, names, metadata):
    """Check destination shard coordinates before any native state is modified."""
    groups = metadata['groups']
    for group, remote in zip(leaf.optimizer.param_groups, groups, strict=True):
        if 'params' in remote or any(group[key] != remote[key] for key in param_group_identifier_keys):
            raise ValueError("TFT optimizer group identifiers differ")
    entries = []
    for (parameter, current), (extent, specs, step) in zip(
        owned_ranges(leaf, names), metadata['entries'], strict=True
    ):
        if current != extent:
            raise ValueError("TFT peer has a different native optimizer shard extent")
        group_index, main = _master(leaf, parameter)
        keys = tuple(key for key, _, _ in specs)
        if keys not in (('param',), ADAM_KEYS):
            raise ValueError("TFT peer supplied invalid Adam state keys")
        if any(shape != main.shape or dtype != main.dtype for _, shape, dtype in specs):
            raise ValueError("TFT peer supplied an incompatible native tensor")
        if step is not None:
            number = step.item() if isinstance(step, torch.Tensor) and step.ndim == 0 else step
            if not isinstance(number, (int, float)) or not math.isfinite(number) or number < 0:
                raise ValueError("TFT peer supplied an invalid Adam step")
        entries.append((main, specs, step, group_index))
    return groups, entries


@torch.no_grad()
def apply_leaf(leaf, stage, values):
    groups, entries = stage
    native_groups = leaf.optimizer.param_groups
    for group, remote in zip(native_groups, groups, strict=True):
        if 'step' not in remote:
            group.pop('step', None)
        for key, value in remote.items():
            if isinstance(value, torch.Tensor) and isinstance(group.get(key), torch.Tensor):
                group[key].copy_(value)
            else:
                group[key] = value
    stream = iter(values)
    for main, specs, step, group_index in entries:
        main.copy_(next(stream))
        optimizer_state = leaf.optimizer.state[main]
        optimizer_state.clear()
        for key, _, _ in specs[1:]:
            optimizer_state[key] = next(stream).to(main.device)
        if step is not None:
            group = native_groups[group_index]
            if isinstance(step, torch.Tensor) and (group.get('capturable') or group.get('fused')):
                step = step.to(main.device)
            optimizer_state['step'] = step
