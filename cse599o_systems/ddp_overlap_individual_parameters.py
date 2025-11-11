import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        Naive DDP wrapper.
        Assumes torch.distributed.init_process_group has already been called
        and the module is already moved to the correct device.
        """
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self._handles = []

        # Broadcast initial parameters from rank 0 to all other ranks
        self._broadcast_parameters()

        # Register gradient hooks to launch async all-reduce
        self._register_grad_hooks()

    def _broadcast_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _grad_hook(self, param: torch.Tensor) -> torch.Tensor:
        """
        Hook that is called when grad is computed for a parameter.
        Launch an async all_reduce on the gradient and stash the handle.
        """
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._handles.append(handle)

    def _register_grad_hooks(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait for all launched all-reduce operations to complete.
        Should be called after backward() and before optimizer.step().
        """
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

        # After synchronization, average the gradients
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= self.world_size

