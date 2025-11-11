from __future__ import annotations
from typing import Any, Dict, Iterable, List, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedStateOptimizer(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | List[Dict[str, Any]],
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ) -> None:
        # Distributed context ---------------------------------------------------
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        self._optimizer_cls = optimizer_cls

        # Ownership bookkeeping -------------------------------------------------
        self._param_owner: Dict[int, int] = {}        # id(param) -> owner rank
        self._all_params_in_order: List[torch.nn.Parameter] = []
        self._global_param_index: int = 0
        self._local_param_groups: List[Dict[str, Any]] = []

        self._wrapped: Optimizer | None = None

        # Call base Optimizer ctor so .param_groups, .defaults, .state exist
        super().__init__(params, defaults=kwargs)

        self._wrapped = self._build_wrapped_optimizer(kwargs)

    def _assign_owner(self, p: torch.nn.Parameter) -> int:
        """Deterministic round-robin owner assignment."""
        pid = id(p)
        if pid in self._param_owner:
            return self._param_owner[pid]

        owner = self._global_param_index % max(self.world_size, 1)
        self._param_owner[pid] = owner
        self._all_params_in_order.append(p)
        self._global_param_index += 1
        return owner

    def _extract_local_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """Take a full param group and keep only params owned by this rank."""
        local_group: Dict[str, Any] = {k: v for k, v in group.items() if k != "params"}
        local_params: List[torch.nn.Parameter] = []
        for p in group["params"]:
            if self._assign_owner(p) == self.rank:
                local_params.append(p)
        local_group["params"] = local_params
        return local_group

    def _build_wrapped_optimizer(self, kwargs: Dict[str, Any]) -> Optimizer | None:
        """Instantiate the real optimizer (e.g., AdamW) on the local shard."""
        non_empty_groups = [g for g in self._local_param_groups if len(g["params"]) > 0]
        if not non_empty_groups:
            return None
        return self._optimizer_cls(non_empty_groups, **kwargs)

    # add_param_group is called by base Optimizer __init__
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Called by base __init__ AND possibly later during training."""

        # Normalize to list to avoid exhausting generators
        raw = param_group.get("params")
        if isinstance(raw, torch.nn.Parameter):
            params_list = [raw]
        else:
            params_list = list(raw)

        full_group = {**param_group, "params": params_list}

        # Build the local shard of this group for the wrapped optimizer
        local_group = self._extract_local_group(full_group)
        self._local_param_groups.append(local_group)

        # Register the full group with the base Optimizer so that
        # zero_grad, state dict, etc. see all parameters.
        super().add_param_group(full_group)

        # If the wrapped optimizer already exists (i.e., this is called
        # AFTER __init__), append this local group there as well.
        if self._wrapped is not None and len(local_group["params"]) > 0:
            self._wrapped.add_param_group(local_group)

    @torch.no_grad()
    def step(self, closure=None, **kwargs: Any):
        """Run local shard step, then broadcast updated params from owners."""
        loss = None
        if self._wrapped is not None:
            if closure is None:
                loss = self._wrapped.step(**kwargs)
            else:
                loss = self._wrapped.step(closure, **kwargs)

        # Synchronize parameters across ranks
        if (
            self.world_size > 1
            and dist.is_available()
            and dist.is_initialized()
        ):
        
            for p in self._all_params_in_order:
                owner = self._param_owner[id(p)]
                dist.broadcast(p.data, src=owner)

        return loss
