import torch
from torch import nn
import geoopt
from typing import Optional, Sequence, Tuple, Callable, Mapping, Iterable
from collections import defaultdict


class SSTNN(nn.Module):
    """Spatial State Transfer Neural Network"""

    def __init__(
        self,
        hidden_size: int,
        input_size: int,
        output_size: int,
        n_channels: int,
        n_dims: int,
        transforms,
        tau,
    ):
        assert all(
            map(
                lambda x: x > 0,
                [hidden_size, input_size, output_size, n_channels],
            )
        ), "Can't initialize this class by passing negative quantities."
        assert (
            n_dims > 1
        ), f"invalid 'n_dims' argument. Cannot map {n_dims}d vectors to {n_dims-1} hypersphere."
        super(SSTNN, self).__init__()
        self.i_size, self.o_size, self.h_size = input_size, output_size, hidden_size
        self.n_neurons = input_size + hidden_size + output_size
        self.__within_input = self.__within_range(input_size)
        self.__within_output = self.__within_range(input_size, input_size + output_size)
        self.state = defaultdict(list)
        self.params = nn.ParameterDict({})
        self.transforms = transforms
        self.tau = [lambda x: x for _ in range(self.n_neurons)] # placeholder
        M = geoopt.Sphere()
        self.pos = nn.ParameterDict(
            {
                "head": geoopt.ManifoldParameter(
                    M.projx(torch.randn(self.n_neurons, n_dims)), M
                ),
                "tail": geoopt.ManifoldParameter(
                    M.projx(torch.randn(self.n_neurons, n_dims)), M
                ),
            }
        )

    def add_to_state(self, x: Mapping[int, torch.Tensor], safe: bool = True) -> None:
        for _index in x:
            try:
                if safe and not self.__within_input(_index):
                    raise IndexError()
                self.state["pos"].append(self.pos["tail"][_index])
                self.state["data"].append(x[_index])
                self.state["tau"].append(self.tau[_index])
            except:
                print(
                    f"Tried adding value to an invalid input index ({_index}). "
                    f"Keys must be indexes in the [0, {self.i_size}) range."
                )

    def _score(self, data_pos: torch.Tensor, heads_pos: torch.Tensor) -> torch.Tensor:
        return data_pos @ heads_pos.T

    def _activate(self, scores: torch.Tensor) -> Iterable[int]:
        return list(range(self.i_size, self.n_neurons))

    def _aggregate(self, data: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Weights and aggregates data to each neuron's head.
        """
        return torch.einsum("ij,kl->lj", data, scores)

    def __within_range(a: int, b: Optional[int] = None) -> bool:
        if b is None:

            def __is_within_range(x):
                return x < a

        else:
            a, b = min(a, b), max(a, b)

            def __is_within_range(x):
                return x >= a and x < b

        return __is_within_range

    def forward(self, x: Optional[Mapping[int, torch.Tensor]] = None) -> dict:
        if x is not None:
            self.add_to_state(x)
        data_pos = torch.stack(self.state["pos"])
        data = torch.stack(self.state["data"])
        scores = self._score(data_pos, self.pos["head"])
        active_neurons_idxs = self._activate(scores)
        scores = torch.stack(
            [self.state["tau"][i](scores[i]) for i in range(scores.shape[0])]
        )
        data = self._aggregate(data, scores)
        out = {}
        self.state.clear()
        for _order, _index in enumerate(active_neurons_idxs):
            data = self.transforms[_index](data[_order], **self.params)
            if self.__within_output(_index):
                out[_index] = data
                continue
            self.state["data"].append(data)
            self.state["pos"].append(self.pos["tail"][_index])
            self.state["tau"].append(self.tau[_index])
        return out
