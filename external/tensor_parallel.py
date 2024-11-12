# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.distributed as dist
import torch
import torch.nn.functional as F

from torch.autograd import Function

import math

from axonn import axonn as ax
from axonn.intra_layer.communication import (
    Drop,
    Gather,
)

from typing import Optional, Sequence


# Wrapper for custom_fwd to handle different versions of PyTorch
def version_aware_custom_fwd(*args, **kwargs):
    version = torch.__version__.split(".")
    major_version = int(version[0])
    minor_version = int(version[1])
    if major_version > 2 or (major_version == 2 and minor_version >= 4):
        # For PyTorch version >= 2.4, pass device_type="cuda"
        return torch.amp.custom_fwd(device_type="cuda")(*args, **kwargs)
    else:
        # For PyTorch version < 2.4, no arguments are required
        return torch.cuda.amp.custom_fwd(*args, **kwargs)


# Wrapper for custom_bwd to handle different versions of PyTorch
def version_aware_custom_bwd(*args, **kwargs):
    version = torch.__version__.split(".")
    major_version = int(version[0])
    minor_version = int(version[1])
    if major_version > 2 or (major_version == 2 and minor_version >= 4):
        # For PyTorch version >= 2.4, pass device_type="cuda"
        return torch.amp.custom_bwd(device_type="cuda")(*args, **kwargs)
    else:
        # For PyTorch version < 2.4, no arguments are required
        return torch.cuda.amp.custom_bwd(*args, **kwargs)


def divide(a, b):
    assert a % b == 0
    return a // b


@torch.no_grad()
def extract_local_params_from_full_params(
    params, out_features_group, in_features_group
):
    params = Drop.apply(params, in_features_group)
    params = Drop.apply(torch.t(params).contiguous(), out_features_group)
    params = torch.t(params).contiguous()
    return params


@torch.no_grad()
def initialize_params(
    out_features,
    in_features,
    out_features_group,
    in_features_group,
    init_method,
    init_device="cuda",
):
    params = torch.empty((out_features, in_features), device=init_device)
    init_method(params)
    params = extract_local_params_from_full_params(
        params, out_features_group, in_features_group
    )
    return params


@torch.no_grad()
def default_init_method(weight):
    return torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))



class TPLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *args,
        transpose=False,
        bias=True,
        skip_bias_add=False,
        init_method=None,
        expert_mode=True,
        tensor_parallel_dims: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super(TPLinear, self).__init__()
        assert expert_mode, "Only expert mode allowed in inference"

        # weights are shaped [out_features, in_features]
        # in_features are distributed across self.inner_group (X tensor parallel group)
        # out_features are distributed across self.inner_group (Y tensor parallel group)
        # if transpose is true then X and Y are swapped
        if tensor_parallel_dims is not None and torch.distributed.get_rank() == 0:
            print(
                "Manually setting TP dims for a layer with shape",
                f" - {(in_features, out_features)} | tp-dims = {tensor_parallel_dims}",
            )
        self.inner_group, self.outer_group, self.depth_group = (
            ax.comm_handle.get_intra_layer_groups(tensor_parallel_dims)
        )
        if transpose:
            self.inner_group, self.outer_group = self.outer_group, self.inner_group

        # depth_group is the Z tensor parallel group (akin to FSDP)
        self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group

        # calculating the sizes of each tensor parallel process group
        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        self.depth_group_size = dist.get_world_size(self.depth_group)

        assert self.depth_group_size == 1

        # these are the in and out features of the full global weight matrix
        self.in_features = in_features
        self.out_features = out_features

        # expert mode = True -> user needs to parallelize non-linear layers manually
        # expert mode = False -> non-linear layers are parallelized using
        #                        data parallelism
        #                        automatically by AxoNN. This does involve some
        #                        extra communication
        #                        at the beginning and end of each linear layer.
        self.expert_mode = expert_mode

        # init_method -> function to initialize the weight matrix
        if init_method is None:
            init_method = default_init_method

        # in_features should be divisible by inner_group_size
        assert in_features % self.inner_group_size == 0
        # in_features should be divisible by inner_group_size
        assert out_features % self.outer_group_size == 0
        # local_in_features - this is the number of in_features on each GPU
        self.local_in_features = divide(in_features, self.inner_group_size)
        # local_out_features - this is the number of out_features on each GPU
        self.local_out_features = divide(out_features, self.outer_group_size)
        # initialize the weight matrix and grab the local slice for each GPU
        initial_params = initialize_params(
            out_features,
            in_features,
            self.outer_group,
            self.inner_group,
            init_method,
        )
        # register the weight matrix as a trainable parameter.
        self.weight = torch.nn.Parameter(initial_params, requires_grad=True)

        # extra book-keeping for the weight tensor.
        # this is needed by AxoNN layer in the sync_gradients and
        # gradient clipping functions.
        setattr(self.weight, "is_tensor_parallel", True)
        setattr(self.weight, "needs_depth_parallel_gradient_sync", False)
        setattr(
            self.weight,
            "process_group_for_norm_reduction",
            ax.comm_handle.intra_layer_group,
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    self.local_out_features,
                )
            )
            setattr(self.bias, "is_tensor_parallel", True)
            setattr(self.bias, "needs_depth_parallel_gradient_sync", True)
            if not transpose:
                setattr(
                    self.bias,
                    "process_group_for_norm_reduction",
                    ax.comm_handle.outer_intra_layer_parallel_group,
                )
            else:
                setattr(
                    self.bias,
                    "process_group_for_norm_reduction",
                    ax.comm_handle.inner_intra_layer_parallel_group,
                )
        else:
            self.bias = None

        self.skip_bias_add = skip_bias_add
        self._old_load_from_state_dict = self._load_from_state_dict
        self._load_from_state_dict = self._modified_load_from_state_dict
        self._old_state_dict = self.state_dict
        self.state_dict = self._modified_state_dict

    #@torch.compiler.disable(recursive=True)
    def all_reduce(self, x):
        dist.all_reduce(x, group=self.inner_group)    
        return x
    
    def matmul(self, w, x):
        return F.linear(x, w)

    #@torch.compiler.disable(recursive=False)
    def forward(
        self,
        x,
        cache_weights_in_all_gather=False,
    ):

        x = self.matmul(self.weight, x)
        x = self.all_reduce(x)

        if self.bias is None:
            return x
        else:
            bias = self.bias
            if self.skip_bias_add:
                return x, bias
            else:
                return x + bias

    def _is_full_weight_matrix(self, weight):
        return (
            weight.ndim == 2
            and weight.size(0) == self.out_features
            and weight.size(1) == self.in_features
        )

    def _is_sharded_weight_matrix(self, weight):
        return weight.ndim == 2 and weight.size(0) == self.local_out_features and weight.size(1) == self.local_in_features

    @torch.no_grad()
    def _modified_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight = (
            state_dict[prefix + "weight"] if prefix + "weight" in state_dict else None
        )

        if weight is not None:
            is_full_weight_matrix = self._is_full_weight_matrix(weight)
            is_sharded_weight_matrix = self._is_sharded_weight_matrix(weight)

            assert (
                is_full_weight_matrix or is_sharded_weight_matrix
            ), "This is neither a full checkpoint nor a sharded checkpoint"

            if is_full_weight_matrix:
                out_features_group, in_features_group = (
                    self.outer_group,
                    self.inner_group,
                )
                weight = extract_local_params_from_full_params(
                    weight, out_features_group, in_features_group
                )

            state_dict[prefix + "weight"] = weight

        if self.bias is not None:
            bias = (
                state_dict[prefix + "bias"] if prefix + "bias" in state_dict else None
            )
            if bias is not None:
                if bias.size(0) == self.out_features:
                    bias = Drop.apply(bias, self.outer_group)
                    state_dict[prefix + "bias"] = bias
                else:
                    assert (
                        bias.size(0) == self.local_out_features
                    ), "This is neither a full checkpoint nor a sharded checkpoint"

        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @torch.no_grad()
    def _modified_state_dict(self, *args, **kwargs):
        local_state_dict = self._old_state_dict(*args, **kwargs)
        weight_key, bias_key = None, None    
        for key in local_state_dict:
            if "weight" in key:
                weight_key = key
            if "bias" in key:
                bias_key = key
        local_weight = local_state_dict[weight_key]
        global_weight = gather_full_params_from_local_params(
            local_weight, self.outer_group, self.inner_group, self.depth_group, (self.local_out_features, self.local_in_features)
        )
        if bias_key is not None:
            local_bias = local_state_dict[bias_key]
            global_bias = Gather.apply(local_bias, self.outer_group) 

        if torch.distributed.get_rank() == 0:
            local_state_dict[weight_key] = global_weight.cpu()
            if bias_key is not None:
                local_state_dict[bias_key] = global_bias
        
        return local_state_dict
