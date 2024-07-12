import torch
from torch import nn
import torch.nn.functional as F
from src.kan import KAN, KANLinear

class DeepQNetwork(nn.Module):
    """ Deep Q Network class for DQN algorithm
    """
    def __init__(self, num_states: int, num_actions: int, num_hidden_units: int=128,
                 model_type: str = "mlp") -> None:
        super(DeepQNetwork, self).__init__()
        """ Initialize DeepQNetwork class
        Args:
            num_states (int): number of states
            num_actions (int): number of actions
            num_hidden_units (int): number of hidden units
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_hidden_units = num_hidden_units
        
        if model_type.lower() == "kan":
            self.nn = KAN([self.num_states, self.num_hidden_units, self.num_actions])
        elif model_type.lower() == "mlp":
            self.nn = nn.Sequential(
                nn.Linear(self.num_states, self.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self.num_hidden_units, self.num_actions)
            )
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

