from typing import List, Optional, Sequence, Tuple, Union

import torch

from ....torch_utility import get_batch_size, get_device
from ....types import TorchObservation
from .base import ContinuousQFunctionForwarder, DiscreteQFunctionForwarder

__all__ = [
    # "DiscreteSeqQFunctionForwarder",
    "ContinuousSeqQFunctionForwarder",
]

def compute_seq_q_function_error(
    forwarders: Union[
        Sequence[DiscreteQFunctionForwarder],
        Sequence[ContinuousQFunctionForwarder],
    ],
    observations: TorchObservation,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    target: torch.Tensor,
    terminals: torch.Tensor,
    gamma: Union[float, torch.Tensor] = 0.99,
) -> torch.Tensor:
    assert target.ndim == 3
    td_sum = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=get_device(observations),
    )
    for n, forwarder in enumerate(forwarders):
        loss = forwarder.compute_error(
            observations=observations,
            actions=actions,
            rewards=rewards[n],
            target=target[n],
            terminals=terminals,
            gamma=gamma,
            reduction="none",
        )
        td_sum += loss.mean()
    return td_sum


class ContinuousSeqQFunctionForwarder:
    _forwarders: Sequence[ContinuousQFunctionForwarder]
    _action_size: int

    def __init__(
        self,
        forwarders: Sequence[ContinuousQFunctionForwarder],
        action_size: int,
    ):
        self._forwarders = forwarders
        self._action_size = action_size

    def compute_expected_q(
        self, x: TorchObservation, action: torch.Tensor
    ) -> torch.Tensor:
        values = []
        for forwarder in self._forwarders:
            value = forwarder.compute_expected_q(x, action)
            values.append(
                value.view(
                    (
                        x[0].shape[0]
                        if isinstance(x, (list, tuple))
                        else x.shape[0]  # type: ignore
                    ),
                    1,
                )
            )
        return torch.stack(values, dim=0)

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
    ) -> torch.Tensor:
        return compute_seq_q_function_error(
            forwarders=self._forwarders,
            observations=observations,
            actions=actions,
            rewards=rewards,
            target=target,
            terminals=terminals,
            gamma=gamma,
        )

    @property
    def forwarders(self) -> Sequence[ContinuousQFunctionForwarder]:
        return self._forwarders
    