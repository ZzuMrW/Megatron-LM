# Adapted from mindspeed_llm/tasks/high_availability/high_availability_helper.py at effa8e78.
import os


def get_env_args(args):
    """Read the official comma-separated TFT deployment switches."""
    strategies = {item.lower() for item in os.getenv('HIGH_AVAILABILITY', '').split(',')}
    args.enable_high_availability = bool(strategies & {'dump', 'recover', 'retry'})
    args.enable_worker_reboot = 'recover' in strategies
    args.enable_hbmfault_repair = 'retry' in strategies
    return args
