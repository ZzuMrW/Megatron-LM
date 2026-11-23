"""Optional MindIO TFT integration for native Megatron training."""

from contextlib import nullcontext

from .high_availability_helper import get_env_args

_started = False
_training = None


def start(args):
    """Start TFT after binding the local device and before creating the world."""
    global _started
    if not get_env_args(args).enable_high_availability:
        return
    from .initialize_patch import initialize_distributed

    initialize_distributed(args)
    _started = True


def enabled():
    return _started


def bootstrapping():
    if not _started:
        return False
    from .tft_arf_group_repair import tft_is_arf_reboot_node

    return tft_is_arf_reboot_node()


def model_setup(recovery):
    if not bootstrapping():
        return nullcontext()
    from .communication_patch import replacement_model_setup

    return replacement_model_setup(recovery)


def run(train, model, optimizer, scheduler, config, checkpointing, build_iterators, iterators, recovery):
    """Register the official callbacks and run the native trainer."""
    global _training
    from .tft_train_initialize import TrainingState, tft_train

    _training = TrainingState(model, optimizer, scheduler, config, checkpointing, build_iterators, iterators, recovery)
    try:
        result = tft_train(train, _training)
        return result, _training.iterators
    finally:
        _training = None


def begin(step):
    if _training is not None:
        _training.begin(step)


def commit(step, flops):
    if _training is not None:
        _training.commit(step, flops)


def backend(default):
    return 'hccl' if enabled() else default


def allow_approximate_sampler_position():
    return _training is None or not _training.restoring


def configure_loader_rng(train_loader, valid_loaders, test_loader):
    if enabled():
        from .dataloader_patch import configure_loader_rng as configure

        configure(train_loader, valid_loaders, test_loader)


def record_log(losses, grad_norm, num_zeros):
    if _training is not None:
        loss = losses.get('lm loss')
        if loss is not None:
            loss = loss.item() if hasattr(loss, 'item') else float(loss)
        _training.log_args = loss, grad_norm, num_zeros
