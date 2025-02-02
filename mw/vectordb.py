import torch
from torch import nn


class VectorDB:
    def __init__(self, env_dims: int, data_dims: int):
        self.env_dims = env_dims
        self.data_dims = data_dims
        # Positions requires grad to flow back to units' tails
        self.positions = torch.empty(env_dims, 0).requires_grad_(True)
        self.data = torch.empty(0, data_dims).requires_grad_(False)

    def add(self, position: torch.Tensor, data: torch.Tensor):
        assert position.shape[0] == data.shape[0]
        assert position.shape[-1] == self.env_dims
        assert data.shape[-1] == self.data_dims
        self.positions = torch.cat([self.positions, position.T], dim=1)
        self.data = torch.cat([self.data, data], dim=0)

    def lookup(self):
        return self.data

    def similarity(self, position: torch.Tensor):
        assert position.shape[-1] == self.env_dims
        return position @ self.positions

    def update_data(self, data: torch.Tensor):
        data = data.T
        assert data.shape == self.data.shape
        self.data = data

    def clear_zeros(self, eps: float = 1e-6, inspect: bool = False):
        if inspect:
            print(
                "VectorDB.clear_zeros -------- Clears small data values from vector db"
            )
            print(
                f"    Had {self.data.shape[0]} {self.data.shape[1]}-d data points at {self.positions.shape[1]} {self.positions.shape[0]}-d positions"
            )
            __before = self.data.shape[0]

        valid_idxs = torch.where(self.data.norm(dim=-1) > eps)[0]
        self.positions = self.positions[:, valid_idxs]
        self.data = self.data[valid_idxs]

        if inspect:
            print(
                f"    Now has {self.data.shape[0]} {self.data.shape[1]}-d data points at {self.positions.shape[1]} {self.positions.shape[0]}-d positions"
            )
            print(f"    Cleared {__before - self.data.shape[0]} data points")
