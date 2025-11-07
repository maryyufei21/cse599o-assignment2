from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
import torch.cuda.nvtx as nvtx

class AdamW(torch.optim.Optimizer):
    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            eps = group['eps']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]
                t = state.get('step', 0) + 1
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                weight_decay = state.get('weight_decay', weight_decay)

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                state['m'] = m
                state['v'] = v
                state['step'] = t
                state['lr'] = lr_t

                p.data = p.data - lr_t * (m / (torch.sqrt(v) + eps))
                p.data = p.data - lr * weight_decay * p.data

        return loss


class AdamWAnnotated(torch.optim.Optimizer):
    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @nvtx.range("AdamWAnnotated step")
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        with nvtx.range("AdamWAnnotated closure"):
            loss = None if closure is None else closure()

        with nvtx.range("AdamWAnnotated param_groups_loop"):
            for group in self.param_groups:
                lr = group['lr']
                weight_decay = group['weight_decay']
                eps = group['eps']
                beta1, beta2 = group['betas']
                
                for p in group['params']:
                    if p.grad is None:
                        continue

                    with nvtx.range("AdamWAnnotated get_state"):
                        grad = p.grad.data
                        state = self.state[p]
                        t = state.get('step', 0) + 1
                        m = state.get('m', torch.zeros_like(p.data))
                        v = state.get('v', torch.zeros_like(p.data))
                        weight_decay = state.get('weight_decay', weight_decay)

                    with nvtx.range("AdamWAnnotated compute_moments"):
                        m = beta1 * m + (1 - beta1) * grad
                        v = beta2 * v + (1 - beta2) * (grad * grad)
                        lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                    
                    with nvtx.range("AdamWAnnotated update_state"):
                        state['m'] = m
                        state['v'] = v
                        state['step'] = t
                        state['lr'] = lr_t

                    with nvtx.range("AdamWAnnotated update_params"):
                        p.data = p.data - lr_t * (m / (torch.sqrt(v) + eps))
                        p.data = p.data - lr * weight_decay * p.data

        return loss
