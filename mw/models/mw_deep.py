import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


class ManifoldWorms(nn.Module):
    """
    Manifold Worms allows for learnable and dynamic neural network
    architectures with $d$ orthogonal directions or up to $d$ layers.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        d: int = 3,
        n_channels: int = 1
    ):
        super(ManifoldWorms, self).__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size
        self.out_size = output_size
        self.n_channels = n_channels
        self.d = d
        self.__transformation_mask = torch.ones(1, output_size + hidden_size, n_channels)
        self.__transformation_mask[:, :output_size] = 0
        self.bias = nn.Parameter(torch.zeros(output_size + hidden_size, n_channels))
        self.positions = nn.ParameterDict(
            {
                "tails": nn.Parameter(
                    torch.randn(input_size + hidden_size, d), requires_grad=True
                ),
                "heads": nn.Parameter(
                    torch.randn(output_size + hidden_size, d), requires_grad=True
                ),
            }
        )
        self.post_step()
        self.clear_state()
        self.register_buffer('_state', self.state)

    def post_step(self, *args, **kwargs):
        """
        Projects updated positions back to the surface of a hypersphere
        and recalculates the weight matrix.
        
        Call after updating positions
        OR
        Pass as a **step post hook** to this module's optimizer (preferred)
        """
        self.normalize_positions()
        self.__similarity_matrix = self.positions["heads"] @ self.positions["tails"].T

    def normalize_positions(self):
        """
        Projects updated positions back to the surface of a hypersphere.
        """
        with torch.no_grad():
            for name in self.positions:
                self.positions[name].data.copy_(
                    F.normalize(self.positions[name].data, p=2, dim=1)
                )

    def clear_state(self):
        """
        Resets the state and batch size. Use with caution.
        """
        self.state = torch.zeros(1, self.in_size + self.hidden_size, self.n_channels)

    def normalize_grads(self, p: int=1, eps: float=1e-6):
        """
        Optional feature for training.
        Performs a parameter-wise normalization of gradients.
        """
        for param in self.parameters():
            if param.grad is not None:
                param.grad.div_(param.grad.norm(p).clip(eps))

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        sigma: float = 0
        ) -> torch.Tensor:
        """
        Each of the *hidden_size* worms "eat" the nearby resources, process them
        and outputs the transformation to another location.
        *input_size* entrances and *output_size* exits provide an interface to an
        otherwise closed dynamical system. 
        """
        # construct weight matrix
        weight = (
            self.__similarity_matrix + torch.randn_like(weight) * sigma
            if sigma > 0 else
            self.__similarity_matrix
        )
        # prep new state tensor to be processed
        new_state = self.state.clone()
        if x is not None:
            new_state = new_state + F.pad(x, (0, 0, 0, self.hidden_size))
        # core transformations
        new_state = weight @ new_state
        new_state = new_state + self.bias
        new_state = F.tanh(new_state)
        # collect output
        y = new_state[:, :self.out_size]
        # updates state
        new_state = new_state * self.__transformation_mask
        new_state = F.pad(new_state, (0, 0, self.in_size - self.out_size, 0))
        self.state = new_state
        return y
