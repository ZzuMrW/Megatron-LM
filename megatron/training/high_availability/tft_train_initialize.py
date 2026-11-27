# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
"""Port the official TFT bootstrap, callback registration and training wrapper."""

import os
import random
from copy import deepcopy

import numpy as np
import torch
from mindio_ttp import framework_ttp as tft

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import unwrap_model
from megatron.training import get_args
from megatron.training.checkpointing import get_rng_state

from .tft_arf_group_repair import (
    arf_rebuild_process_group_callback,
    tft_is_arf_reboot_node,
    update_arf_reboot_flag,
)
from .tft_dump import tft_rename_callback, tft_save_callback
from .tft_optimizer_data_repair import repair_callback
from .tft_replica_group import optimizer_domains
from .tft_rollback import rollback_callback
from .tft_stop_clean import clean_callback, stop_callback, torch_sync
from .tft_tensor_recovery import model_tensors


def tft_init_controller_processor():
    args = get_args()
    default_ip = '127.0.0.1'
    ttp_ip = os.getenv('TTP_ADDR', default_ip)
    controller_ip = os.getenv('CONTROLLER_ADDR', default_ip)
    if controller_ip == default_ip:
        controller_ip = ttp_ip
    processor_ip = os.getenv('PROCESSOR_ADDR', default_ip)
    if processor_ip == default_ip:
        processor_ip = ttp_ip
    port = 8000

    os.environ['TTP_STOP_CLEAN_BEFORE_DUMP'] = '1'
    start_controller = args.rank == 0 and os.getenv('MINDX_TASK_ID') is None
    if start_controller:
        tft.tft_init_controller(args.rank, args.world_size, False, args.enable_worker_reboot, False)
        tft.tft_start_controller(controller_ip, port, False, '')
    tft.tft_init_processor(
        args.rank, args.world_size, False, False, '', args.enable_hbmfault_repair, args.enable_worker_reboot, False
    )
    tft.tft_start_processor(processor_ip, port)
    if tft.tft_is_reboot_node():
        if not args.enable_worker_reboot or tft.tft_get_reboot_type() != 'arf':
            raise RuntimeError("TFT reported a replacement outside the ARF deployment mode")
        update_arf_reboot_flag(True)


def tft_register_processor(training):
    args = get_args()
    replica_info = [
        {'rank_list': list(ranks), 'replica_cnt': args.num_distributed_optimizer_instances, 'replica_shift': 0}
        for ranks, _, _ in training.domains
    ]
    tft.tft_set_optimizer_replica(args.rank, replica_info)
    plain_dp = torch.distributed.get_process_group_ranks(mpu.get_data_parallel_group())
    tft.tft_set_dp_group_info(args.rank, plain_dp)

    # In generic PyTorch mode the SDK wraps positional arguments in a list.
    tft.tft_register_save_ckpt_handler(lambda step, info, argv, ctx: tft_save_callback(step, info, argv[0], ctx))
    tft.tft_register_rename_handler(lambda step, argv: tft_rename_callback(step, argv[0]))
    tft.tft_register_stop_handler(lambda argv, ctx: stop_callback(argv[0], ctx))
    tft.tft_register_clean_handler(lambda uce, argv, ctx: clean_callback(uce, argv[0], ctx))
    tft.tft_register_repair_handler(
        lambda step, rebuild, errors, info, argv, ctx: repair_callback(step, rebuild, errors, info, argv[0], ctx)
    )
    tft.tft_register_rollback_handler(lambda step, argv, ctx: rollback_callback(step, argv[0], ctx))
    tft.tft_register_rebuild_group_handler(
        lambda fault_ranks, argv, ctx: arf_rebuild_process_group_callback(fault_ranks, argv[0], ctx)
    )
    tft.tft_register_stream_sync_handler(torch_sync)


def tft_train(train, training):
    from .communication_patch import seal_group_inventory

    seal_group_inventory()
    tft_register_processor(training)
    first_call = True

    @tft.tft_exception_handler
    def wrapped(state):
        nonlocal first_call
        if first_call:
            first_call = False
            if tft_is_arf_reboot_node():
                tft.tft_report_error(tft.ReportState.RS_INIT_FINISH.value)
                if tft.tft_wait_next_action() != tft.Action.RETRY.value or not state.rebuilt:
                    raise ValueError("TFT has not completed ARF communicator recovery")
                raise RuntimeError("ARF FINISH")
            state.initialize_step()
        if state.damaged:
            raise ValueError("TFT has not restored the paused optimizer")
        return train(*state.iterators[:2])

    return wrapped(training)


class TrainingState:
    """Hold the position and handles required by the official callback sequence."""

    counters = ('consumed_train_samples', 'skipped_train_samples', 'consumed_valid_samples')

    def __init__(self, model, optimizer, scheduler, config, checkpointing, build_iterators, iterators, recovery):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.config, self.checkpointing = config, checkpointing
        self.build_iterators, self.iterators = build_iterators, iterators
        self.recovery = recovery
        self.domains = optimizer_domains(optimizer, get_args())
        self.names = {}
        for chunk, module in enumerate(unwrap_model(model)):
            for name, parameter in module.named_parameters(remove_duplicate=False):
                self.names.setdefault(parameter, []).append((chunk, name))
        self.names = {parameter: tuple(names) for parameter, names in self.names.items()}
        self.sync_hooks = {key: getattr(config, key) for key in ('no_sync_func', 'grad_sync_func', 'param_sync_func')}
        self.step = None
        self.pending = None
        self.progress = None
        self.received = None
        self.loaded = False
        self.damaged = tft_is_arf_reboot_node()
        self.rebuilt = False
        self.restoring = False
        self.log_args = (None, None, None)
        self.repaired_log_args = None

    def _progress(self, step, flops):
        args = get_args()
        return {
            'step': step,
            'rank': args.rank,
            'flops': flops,
            'counters': {key: getattr(args, key) for key in self.counters},
            'scheduler': deepcopy(self.scheduler.state_dict()),
            'rng': deepcopy(get_rng_state('torch', None, None)[0]),
            'model_buffers': {key: value.detach().to('cpu', copy=True) for key, value in model_tensors(self.model)},
        }

    def initialize_step(self):
        args = get_args()
        self.step = args.iteration
        self.progress = self._progress(self.step, args.num_floating_point_operations_so_far)
        if self.step > 0:
            tft.tft_start_updating_os()
            tft.tft_end_updating_os(self.step)

    def begin(self, step):
        if self.pending is not None or self.damaged:
            raise RuntimeError("TFT cannot start an update while native state is incomplete")
        tft.tft_start_updating_os()
        self.pending = step

    def commit(self, step, flops):
        if step != self.pending:
            raise RuntimeError("TFT step does not match the active optimizer transaction")
        torch_sync()
        self.progress = self._progress(step, flops)
        self.step, self.pending = step, None
        args = get_args()
        args.iteration = args.curr_iteration = step
        args.num_floating_point_operations_so_far = flops
        tft.tft_end_updating_os(step)
        tft.tft_pause_train(step)

    def require_state(self, step):
        if step <= 0 or self.step != step or self.pending is not None or self.damaged:
            raise ValueError(f"TFT action requires valid native state at completed step {step}")

    def original_rank_progress(self, step):
        rank = get_args().rank
        progress = self.progress
        if progress is None or progress['step'] != step:
            progress = self.recovery.rank_progress(rank, step) if self.recovery is not None else None
        if progress is None or (progress['rank'], progress['step']) != (rank, step):
            raise ValueError(f"TFT needs completed-step progress for rank {rank} at step {step}")
        return progress


def restore_rng(rng):
    random.setstate(rng['random_rng_state'])
    np.random.set_state(rng['np_rng_state'])
    torch.set_rng_state(rng['torch_rng_state'])
    torch.cuda.set_rng_state(rng['cuda_rng_state'])
    tensor_parallel.get_cuda_rng_tracker().set_states(rng['rng_tracker_states'])
