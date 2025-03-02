import torch
from torch import nn
import geoopt
from typing import Sequence, Optional


class SphericalEmbeddings(nn.Module):

    def __init__(
        self,
        n_dims: int,
        position_keys: Sequence[str],
        learnable_keys: Sequence[str],
        static_keys: Sequence[str],
    ):
        assert len(set(position_keys).union(learnable_keys).union(static_keys)) == len(
            position_keys
        ) + len(learnable_keys) + len(
            static_keys
        ), "All keys must be unique for the same SphericalEmbeddings instance."
        super(SphericalEmbeddings, self).__init__()
        self.d = n_dims
        M = geoopt.Sphere()
        self.__positions = nn.ParameterDict(
            {
                key: geoopt.ManifoldParameter(torch.empty(0, n_dims), M)
                for key in position_keys
            }
        )
        self.__learnable = nn.ParameterDict(
            {key: nn.Parameter(torch.empty(0, 1)) for key in learnable_keys}
        )
        self.__static = {key: [] for key in static_keys}
        self.__key_ref = {}
        self.__key_ref.update({key: self.__positions[key] for key in position_keys})
        self.__key_ref.update({key: self.__learnable[key] for key in learnable_keys})
        self.__key_ref.update({key: self.__static[key] for key in static_keys})

    def __getitem__(self, key, index: Optional[Union[int, Sequence[int]]] = None):
        if index is not None:
            return self.__key_ref[key][index]
        return self.__key_ref[key]
