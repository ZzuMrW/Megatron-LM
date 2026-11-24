# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import time
from logging import getLogger

import torch
import torch_npu
from mindio_ttp.framework_ttp.ttp_decorator import tft_get_repair_type

from .tft_tensor_recovery import model_tensors, update_optimizer_tensors_to_safe
from .utils import ha_constant

ttp_logger = getLogger(__name__)


def stop_callback(training, ctx):
    # stop and clean device
    device = torch.npu.current_device()
    ret = torch_npu.npu.stop_device(device)
    if ret is not None and ret != ha_constant.RET_OK:
        raise RuntimeError("stop failed,end stop callback")


def clean_callback(is_uce_error: bool, training, ctx):
    """
    this function do:
    1) get UCE check result from torch_npu
    2) do some clear before rebuild (avoid OOM) when the check result is UCE_HIGH_LEVEL
    3) HCCL resume and restart device
    """
    device = torch.npu.current_device()
    rank = torch.distributed.get_rank()
    ret = ha_constant.RET_OK
    start_time = time.time()
    unset_gather_handle(training.model)
    if is_uce_error:
        check_memory_result = torch_npu.npu.check_uce_in_memory(device)
        ttp_logger.info(f"rank {rank} check uce memory result: {check_memory_result}")
        if check_memory_result == ha_constant.UCE_LOW_LEVEL:  # no need rebuild
            ret = ha_constant.RET_NO_REBUILD
        elif check_memory_result == ha_constant.UCE_HIGH_LEVEL:  # need rebuild
            training.damaged = True
            update_optimizer_tensors_to_safe(training.optimizer)
            model_update_npu_tensor_to_safe(training.model)
            ret = ha_constant.RET_OK
        else:  # exit
            ret = ha_constant.RET_ERROR
            ttp_logger.error(f"rank {rank} check uce memory result {ret} is abnormal, exiting...")

    clean_type = tft_get_repair_type()
    if clean_type == "retry":
        torch.distributed.reinit_process_group(group=None, rebuild_link=False)
    torch.npu.restart_device(device)

    ttp_logger.info(f'[clean] rank:{rank}, type:{clean_type}, cost:{time.time() - start_time:.3f}s, ret:{ret}')
    return ret


def model_update_npu_tensor_to_safe(models):
    from torch_npu.npu._recovery import (
        update_npu_tensor_to_safe as update_tensor_to_safe,
    )

    for _, tensor in model_tensors(models):
        if tensor.device.type == 'npu':
            update_tensor_to_safe(tensor)
    for model in models:
        for buffer in model.buffers + model.expert_parallel_buffers:
            for bucket in buffer.buckets:
                for tensor in (bucket.param_data, bucket.grad_data):
                    if tensor is not None:
                        update_tensor_to_safe(tensor)


def unset_gather_handle(models):
    for model in models:
        for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
            bucket_group.grad_reduce_handle = None
            bucket_group.param_gather_handle = None
            bucket_group.param_gather_dispatched = False


def torch_sync():
    rank = torch.distributed.get_rank()
    torch.cuda.synchronize()
    ttp_logger.debug(f"[Pause] rank: {rank} finish synchronize")
