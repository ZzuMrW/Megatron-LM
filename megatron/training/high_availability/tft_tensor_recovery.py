# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""The official optimizer safe-storage traversal adapted to native DistOpt."""

import torch


def update_optimizer_tensors_to_safe(optimizer):
    from torch_npu.npu._recovery import update_npu_tensor_to_safe

    visited = set()

    def update(value):
        if isinstance(value, torch.Tensor) and value.device.type == 'npu':
            address = value.untyped_storage().data_ptr()
            if address not in visited:
                visited.add(address)
                update_npu_tensor_to_safe(value)

    for leaf in getattr(optimizer, 'chained_optimizers', [optimizer]):
        update(leaf._scale_one)
        for group in leaf.optimizer.param_groups:
            for parameter in group['params']:
                update(parameter)
            for key, value in group.items():
                if key != 'params':
                    update(value)
        for state in leaf.optimizer.state.values():
            for value in state.values():
                update(value)


def model_tensors(models):
    """Find registered buffers plus native rotary frequencies held outside state_dict."""
    from megatron.core.models.common.embeddings.rotary_pos_embedding import (
        MultimodalRotaryEmbedding,
        RotaryEmbedding,
    )
    from megatron.core.utils import unwrap_model

    for index, model in enumerate(unwrap_model(models)):
        for name, value in model.named_buffers():
            yield (index, name, ''), value
        for name, module in model.named_modules():
            if isinstance(module, (RotaryEmbedding, MultimodalRotaryEmbedding)):
                for field in ('inv_freq', 'inv_freq_extra', 'inv_freq_inter'):
                    value = vars(module).get(field)
                    if isinstance(value, torch.Tensor):
                        yield (index, name, field), value


def restore_model_tensors(models, values):
    """Restore the original rank's completed-step persistent tensor values."""
    from megatron.core.utils import unwrap_model

    tensors = dict(model_tensors(models))
    if tensors.keys() != values.keys():
        raise ValueError("TFT saved different native model-buffer ownership")
    for key, target in tensors.items():
        source = values[key]
        if target.shape != source.shape or target.dtype != source.dtype:
            raise ValueError(f"TFT model buffer layout differs for {key}")
        target.copy_(source)
    for model in unwrap_model(models):
        for module in model.modules():
            clear = getattr(module.forward, 'cache_clear', None)
            if clear is not None:
                clear()
