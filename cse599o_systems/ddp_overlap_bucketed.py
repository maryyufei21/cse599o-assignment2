import torch
import torch.distributed as dist
from typing import List, Dict

class DDPOverlapBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = int(max(1.0, bucket_size_mb) * 1024 * 1024)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        self._buckets: List[Dict] = []
        self._param_to_bucket: Dict[torch.Tensor, Dict] = {}
        self._handles: List[Dict] = [] # [{"handle": ..., "bucket": ..., "packed_grad": ...}]
        self._broadcast_parameters()

        self._build_buckets()

        self._register_grad_hooks()
    
    def _broadcast_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def _build_buckets(self):
        """
        Create buckets of up to bucket_cap_bytes.
        Reverse order (approx. ready order), keep only requires_grad=True
        """
        params_rev = list(reversed(list(self.module.parameters())))
        # print(f"[DEBUG] Rank {self.rank} total param number: {len(params_rev)}")
        current_bucket = {"id": len(self._buckets), "params": [], "size_bytes": 0, "pending": 0}
        self._buckets.append(current_bucket)
        for param in params_rev:
            if not param.requires_grad:
                continue
            
            param_size = param.numel() * param.element_size()
            if param_size + current_bucket["size_bytes"] > self.bucket_size_bytes:
                # Start a new bucket
                current_bucket = {"id": len(self._buckets), "params": [], "size_bytes": 0, "pending": 0}
                self._buckets.append(current_bucket)
            current_bucket["params"].append(param)
            current_bucket["size_bytes"] += param_size
            current_bucket["pending"] += 1
            self._param_to_bucket[param] = current_bucket
        
        # print(f"[DEBUG] Rank {self.rank} built {len(self._buckets)} buckets.")
        # print bucket details
        # for bucket in self._buckets:
        #     print(f"[DEBUG] Rank {self.rank} Bucket {bucket['id']}: {len(bucket['params'])} params, {bucket['size_bytes']} bytes.")
    
    def _pack_bucket(self, bucket: Dict) -> torch.Tensor:
        """
        Given a bucket dict, pack its parameters' gradients into a single tensor.
        """
        grads = [param.grad.data.view(-1) for param in bucket["params"]]
        return torch.cat(grads)

    def _launch_bucket_allreduce(self, bucket: Dict):
        """
        Launch an async all-reduce on the packed gradients of the bucket.
        Stash the handle for later synchronization.
        """
        packed_grad = self._pack_bucket(bucket)
        handle = dist.all_reduce(packed_grad, op=dist.ReduceOp.SUM, async_op=True)
        # print(f"[DEBUG] Rank {self.rank} launched all-reduce for bucket {bucket['id']}.")
        self._handles.append({"handle": handle, "bucket": bucket, "packed_grad": packed_grad})    
        
    def _grad_hook(self, param: torch.Tensor) -> torch.Tensor:
        """
        Hook that is called when grad is computed for a parameter.
        Launch an async all_reduce on the bucket if all params in the bucket
        have their gradients ready.
        """
        bucket = self._param_to_bucket[param]
        bucket["pending"] -= 1
        if bucket["pending"] == 0:
            # All params in this bucket have their gradients ready
            self._launch_bucket_allreduce(bucket)

    def _register_grad_hooks(self):
        """
        Register gradient hooks for all parameters in the module.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._grad_hook)

    def _unpack_bucket(self, bucket: Dict, packed_grad: torch.Tensor):
        """
        Given a bucket and its packed gradient tensor, unpack the gradients
        back into the individual parameters.
        """
        offset = 0
        bucket["pending"] = len(bucket["params"])  # reset pending for next iteration
        for param in bucket["params"]:
            numel = param.numel()
            param.grad.copy_(packed_grad[offset:offset + numel].view_as(param))
            offset += numel
    
    def finish_gradient_synchronization(self):
        """
        Wait for all launched all-reduce operations to complete.
        Should be called after backward() and before optimizer.step().
        """
        for entry in self._handles:
            entry["handle"].wait()
            self._unpack_bucket(entry["bucket"], entry["packed_grad"])
            

        self._handles.clear()

        # After synchronization, average the gradients
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= self.world_size