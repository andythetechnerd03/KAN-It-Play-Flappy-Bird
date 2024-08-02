"""
Code courtesy of FasterKAN by AthanasiosDelis - https://github.com/AthanasiosDelis/faster-kan/tree/master/fasterkan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *
from torch.autograd import Function
from ..kan.feature_extractor import EnhancedFeatureExtractor
from ..kan.fasterkan_layers import FasterKANLayer

class FasterKAN(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -1.2,
        grid_max: float = 1.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
        #use_base_update: bool = True,
        base_activation = None,
        spline_weight_init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            FasterKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent = exponent,
                inv_denominator = inv_denominator,
                train_grid = train_grid ,
                train_inv_denominator = train_inv_denominator,
                #use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
        #print(f"FasterKAN layers_hidden[1:] shape: ", len(layers_hidden[1:]))   
        #print(f"FasterKAN layers_hidden[:-1] shape: ", len(layers_hidden[:-1]))  
        #print("FasterKAN zip shape: \n", *[(in_dim, out_dim) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])]) 
   
        #print(f"FasterKAN self.faster_kan_layers shape: \n", len(self.layers))
        #print(f"FasterKAN self.faster_kan_layers: \n", self.layers)
    
    def forward(self, x):
        for layer in self.layers:
            #print("FasterKAN layer: \n", layer)
            #print(f"FasterKAN x shape: {x.shape}")
            x = layer(x)
        return x

class FasterKANvolver(nn.Module):
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -1.2,
        grid_max: float = 0.2,
        num_grids: int = 8,
        exponent: int = 2,
        inv_denominator: float = 0.5,
        train_grid: bool = False,        
        train_inv_denominator: bool = False,
        #use_base_update: bool = True,
        base_activation = None,
        spline_weight_init_scale: float = 1.0,
        view = [-1, 1, 28, 28],
    ) -> None:
        super(FasterKANvolver, self).__init__()
        
        self.view = view
        # Feature extractor with Convolutional layers
        self.feature_extractor = EnhancedFeatureExtractor(colors = view[1])
        """
        nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1 input channel (grayscale), 16 output channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        """
        # Calculate the flattened feature size after convolutional layers
        flat_features = 256 # XX channels, image size reduced to YxY
        
        # Update layers_hidden with the correct input size from conv layers
        layers_hidden = [flat_features] + layers_hidden
        #print(f"FasterKANvolver layers_hidden shape: \n", layers_hidden)
        #print(f"FasterKANvolver layers_hidden[1:] shape: ", len(layers_hidden[1:]))   
        #print(f"FasterKANvolver layers_hidden[:-1] shape: ", len(layers_hidden[:-1]))   
        #print("FasterKANvolver zip shape: \n", *[(in_dim, out_dim) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])])         
        
        # Define the FasterKAN layers
        self.faster_kan_layers = nn.ModuleList([
            FasterKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                exponent=exponent,
                inv_denominator = 0.5,
                train_grid = False,        
                train_inv_denominator = False,
                #use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])   
        #print(f"FasterKANvolver self.faster_kan_layers shape: \n", len(self.faster_kan_layers))
        #print(f"FasterKANvolver self.faster_kan_layers: \n", self.faster_kan_layers)

    def forward(self, x):
        # Reshape input from [batch_size, 784] to [batch_size, 1, 28, 28] for MNIST [batch_size, 1, 32, 32] for C
        #print(f"FasterKAN x view shape: {x.shape}")
        # Handle different input shapes based on the length of view
        x = x.view(self.view[0], self.view[1], self.view[2], self.view[3])
        #print(f"FasterKAN x view shape: {x.shape}")
        # Apply convolutional layers
        #print(f"FasterKAN x view shape: {x.shape}")
        x = self.feature_extractor(x)
        #print(f"FasterKAN x after feature_extractor shape: {x.shape}")
        x = x.view(x.size(0), -1)  # Flatten the output from the conv layers
        #rint(f"FasterKAN x shape: {x.shape}")
        
        # Pass through FasterKAN layers
        for layer in self.faster_kan_layers:
            #print("FasterKAN layer: \n", layer)
            #print(f"FasterKAN x shape: {x.shape}")
            x = layer(x)
            #print(f"FasterKAN x shape: {x.shape}")
        
        return x