import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .modules import ResidualModule
from .vectordb import VectorDB
from . import influence
from typing import Tuple, Union


class ManifoldWorms(nn.Module):
    def __init__(
        self,
        channel_size: int,
        env_dims: int,
        input_size: int,
        output_size: int,
        n_units: int,
        reach: float = 1,
        garbage_decay: float = 0.9,
    ):
        super(ManifoldWorms, self).__init__()
        self.n_units = n_units
        self.input_size = input_size
        self.output_size = output_size
        self.channel_size = channel_size
        self.env_dims = env_dims
        self.reach_threshold = np.clip(1 - reach, -1, 1).item()
        self.garbage_scale = np.clip(1 - garbage_decay, 0, 1).item()
        self.units = nn.ModuleList(
            [ResidualModule(channel_size) for _ in range(n_units)]
        )
        self.positions = nn.ParameterDict(
            {
                "input_tails": nn.Parameter(
                    torch.randn(input_size, env_dims), requires_grad=True
                ),
                "exit_heads": nn.Parameter(
                    torch.randn(output_size, env_dims), requires_grad=True
                ),
                "unit_heads": nn.Parameter(
                    torch.randn(n_units, env_dims), requires_grad=True
                ),
                "unit_tails": nn.Parameter(
                    torch.randn(n_units, env_dims), requires_grad=True
                ),
            }
        )
        self.db = VectorDB(env_dims, channel_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        with torch.no_grad():
            self.normalize_positions()

    def forward(self, state: torch.Tensor):
        assert state.shape[0] == self.input_size
        assert state.shape[1] == self.channel_size
        self.db.add(self.positions["input_tails"], state)

    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self.normalize_positions()
        similarities = self.db.similarity(
            torch.cat([self.positions["unit_heads"], self.positions["exit_heads"]])
            if self.n_units > 0
            else self.positions["exit_heads"]
        )

        influences = influence.great_distance(similarities, self.reach_threshold)
        data = self.db.lookup()
        distributed_data = influences @ data

        garbage = influences.sum(0).add(-1) * -data.T
        exit_outputs = distributed_data[self.n_units :]

        loss = F.mse_loss(exit_outputs, torch.ones_like(exit_outputs)) + garbage.sum()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        print("Loss", loss.item())

        self.db.update_data(garbage * self.garbage_scale)
        if self.n_units > 0:
            units_outputs = torch.stack(
                [
                    unit(unit_input)
                    for unit, unit_input in zip(self.units, distributed_data)
                ],
                dim=0,
            )
            self.db.add(self.positions["unit_tails"], units_outputs)
        self.db.clear_zeros()

    def normalize_positions(self):
        for name in self.positions:
            self.positions[name].data.copy_(
                F.normalize(self.positions[name], p=2, dim=1).data
            )
