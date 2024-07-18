import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
from torch.autograd import Function

class RSWAFFunction(Function):
    @staticmethod
    def forward(ctx, input, grid, inv_denominator, train_grid, train_inv_denominator):
        # Compute the forward pass
        #print('\n')
        #print(f"Forward pass - grid: {(grid[0].item(),grid[-1].item())}, inv_denominator: {inv_denominator.item()}")

        #print(f"grid.shape: {grid.shape }")
        #print(f"grid: {(grid[0],grid[-1]) }")
        #print(f"inv_denominator.shape: {inv_denominator.shape }")
        #print(f"inv_denominator: {inv_denominator }")
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator)
        tanh_diff = torch.tanh(diff)
        tanh_diff_deriviative = -tanh_diff.mul(tanh_diff) + 1  # sech^2(x) = 1 - tanh^2(x)
        
        # Save tensors for backward pass
        ctx.save_for_backward(input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator)
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator
        
        return tanh_diff_deriviative

##### SOS NOT SURE HOW grad_inv_denominator, grad_grid ARE CALCULATED CORRECTLY YET
##### MUST CHECK https://github.com/pytorch/pytorch/issues/74802
##### MUST CHECK https://www.changjiangcai.com/studynotes/2020-10-18-Custom-Function-Extending-PyTorch/
##### MUST CHECK https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
##### MUST CHECK https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
##### MUST CHECK https://gist.github.com/Hanrui-Wang/bf225dc0ccb91cdce160539c0acc853a

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, tanh_diff, tanh_diff_deriviative, diff, inv_denominator = ctx.saved_tensors
        grad_grid = None
        grad_inv_denominator = None
        
        #print(f"tanh_diff_deriviative shape: {tanh_diff_deriviative.shape }")
        #print(f"tanh_diff shape: {tanh_diff.shape }")
        #print(f"grad_output shape: {grad_output.shape }")
        
        # Compute the backward pass for the input
        grad_input = -2 * tanh_diff * tanh_diff_deriviative * grad_output
        #print(f"Backward pass 1 - grad_input: {(grad_input.min().item(), grad_input.max().item())}")
        #print(f"grad_input shape: {grad_input.shape }")
        #print(f"grad_input.sum(dim=-1): {grad_input.sum(dim=-1).shape}")
        grad_input = grad_input.sum(dim=-1).mul(inv_denominator)
        #print(f"Backward pass 2 - grad_input: {(grad_input.min().item(), grad_input.max().item())}")
        #print(f"grad_input: {grad_input}")
        #print(f"grad_input shape: {grad_input.shape }")
        
        # Compute the backward pass for grid
        if ctx.train_grid:
            #print('\n')
            #print(f"grad_grid shape: {grad_grid.shape }")
            grad_grid = -inv_denominator * grad_output.sum(dim=0).sum(dim=0)#-(inv_denominator * grad_output * tanh_diff_deriviative).sum(dim=0) #-inv_denominator * grad_output.sum(dim=0).sum(dim=0)
            #print(f"Backward pass - grad_grid: {(grad_grid[0].item(),grad_grid[-1].item())}")
            #print(f"grad_grid.shape: {grad_grid.shape }")
            #print(f"grad_grid: {(grad_grid[0],grad_grid[-1]) }")
            #print(f"inv_denominator shape: {inv_denominator.shape }")
            #print(f"grad_grid shape: {grad_grid.shape }")

        # Compute the backward pass for inv_denominator        
        if ctx.train_inv_denominator:
            grad_inv_denominator = (grad_output* diff).sum() #(grad_output * diff * tanh_diff_deriviative).sum() #(grad_output* diff).sum() 
            #print(f"Backward pass - grad_inv_denominator: {grad_inv_denominator.item()}")
            #print(f"diff shape: {diff.shape }")

            #print(f"grad_inv_denominator shape: {grad_inv_denominator.shape }")
            #print(f"grad_inv_denominator : {grad_inv_denominator }")

        return grad_input, grad_grid, grad_inv_denominator, None, None # same number as tensors or parameters



class ReflectionalSwitchFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.train_grid = torch.tensor(train_grid, dtype=torch.bool)
        self.train_inv_denominator = torch.tensor(train_inv_denominator, dtype=torch.bool) 
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        #print(f"grid initial shape: {self.grid.shape }")
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator, dtype=torch.float32), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def forward(self, x):
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator, self.train_grid, self.train_inv_denominator)


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)  # Using Xavier Uniform initialization