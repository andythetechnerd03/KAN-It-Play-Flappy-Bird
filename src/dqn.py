import torch
from torch import nn
import torch.nn.functional as F
from typing import List

from src.kan.kan import KAN, KANLinear
from src.kan.fasterkan import FasterKAN, FasterKANvolver

class DeepQNetwork(nn.Module):
    """ Deep Q Network class for DQN algorithm
    """
    def __init__(self, num_states: int, num_actions: int, hidden_units: List[int],
                 model_type: str = "mlp") -> None:
        super(DeepQNetwork, self).__init__()
        """ Initialize DeepQNetwork class
        Args:
            num_states (int): number of states
            num_actions (int): number of actions
            hidden_units (List[int]): array of hidden units. For example: [128,64] means
                one hidden layer with 128 units followed by one hidden layer with 64 units.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_units = hidden_units
        assert isinstance(hidden_units, list), "hidden_units must be a list of hidden units"

        params_arr = [self.num_states] + hidden_units + [self.num_actions]
        if model_type.lower() == "effkan":
            self.nn = KAN(params_arr)
        elif model_type.lower() == "fasterkan":
            self.nn = FasterKAN(params_arr)
        elif model_type.lower() == "mlp":
            modules = []
            # Add linear layers and ReLU (except the last layer)
            for i in range(len(params_arr)-1):
                modules.append(nn.Linear(params_arr[i], params_arr[i+1]))
                if i != len(params_arr)-2: modules.append(nn.ReLU())
            self.nn = nn.Sequential(*modules)
        else:
            raise ValueError("Model type not supported, please use 'mlp' or 'kan'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor
        """
        return self.nn(x)
    
if __name__ == "__main__":
    # Test DeepQNetwork class
    num_states = 4
    num_actions = 2
    num_hidden_units = 128
    # Test Linear
    net = DeepQNetwork(num_states, num_actions, num_hidden_units)
    x = torch.rand(1, num_states)
    print(net(x), net(x).shape)

    # Test KAN
    num_hidden_units = 8
    net = DeepQNetwork(num_states, num_actions, num_hidden_units, model_type="kan")
    x = torch.rand(1, num_states)
    print(net(x), net(x).shape)

