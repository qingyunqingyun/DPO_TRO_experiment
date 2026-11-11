"""Provide OpenRLHF's differentiable all-gather when flash-attn is unavailable."""

import sys
import types
from importlib.machinery import ModuleSpec

import torch


def install_flash_attn_distributed_fallback():
    try:
        from flash_attn.utils.distributed import all_gather  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        pass

    class AllGather(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input_tensor, process_group):
            ctx.process_group = process_group
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return input_tensor
            world_size = torch.distributed.get_world_size(process_group)
            if world_size == 1:
                return input_tensor
            output = [torch.empty_like(input_tensor) for _ in range(world_size)]
            torch.distributed.all_gather(output, input_tensor.contiguous(), group=process_group)
            return torch.cat(output, dim=0)

        @staticmethod
        def backward(ctx, grad_output):
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return grad_output, None
            world_size = torch.distributed.get_world_size(ctx.process_group)
            if world_size == 1:
                return grad_output, None
            rank = torch.distributed.get_rank(ctx.process_group)
            chunks = grad_output.chunk(world_size, dim=0)
            return chunks[rank].contiguous(), None

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__path__ = []
    flash_attn.__spec__ = ModuleSpec("flash_attn", loader=None, is_package=True)
    utils = types.ModuleType("flash_attn.utils")
    utils.__path__ = []
    utils.__spec__ = ModuleSpec("flash_attn.utils", loader=None, is_package=True)
    distributed = types.ModuleType("flash_attn.utils.distributed")
    distributed.__spec__ = ModuleSpec("flash_attn.utils.distributed", loader=None)
    distributed.all_gather = AllGather.apply
    sys.modules.setdefault("flash_attn", flash_attn)
    sys.modules.setdefault("flash_attn.utils", utils)
    sys.modules.setdefault("flash_attn.utils.distributed", distributed)
