# Adapted from mindspeed_llm/tasks/high_availability/dataloader_patch.py at effa8e78.
"""Rebuild the native iterator at the restored sample cursor."""


def build_dataset(training):
    """Recreate train and validation iterators using Megatron's builder."""
    training.restoring = True
    try:
        training.iterators = training.build_iterators()
    finally:
        training.restoring = False


def configure_loader_rng(train_loader, valid_loaders, test_loader):
    """Keep native iterator construction from advancing the restored training RNG."""
    import torch

    from megatron.training import get_args

    for index, loader in enumerate((train_loader, *(valid_loaders or ()), test_loader)):
        if loader is not None:
            loader.generator = torch.Generator().manual_seed(get_args().seed + index)
