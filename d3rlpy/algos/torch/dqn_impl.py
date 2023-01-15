import copy
from typing import Optional

import numpy as np
import torch
from torch.optim import Optimizer

from ...dataset import Shape
from ...models.builders import create_discrete_q_function
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import EnsembleDiscreteQFunction, EnsembleQFunction
from ...preprocessing import ObservationScaler, RewardScaler
from ...torch_utility import (
    TorchMiniBatch,
    hard_sync,
    to_device,
    torch_api,
    train_api,
)
from .base import TorchImplBase
from .utility import DiscreteQFunctionMixin

__all__ = ["DQNImpl", "DoubleDQNImpl"]


class DQNImpl(DiscreteQFunctionMixin, TorchImplBase):

    _learning_rate: float
    _optim_factory: OptimizerFactory
    _encoder_factory: EncoderFactory
    _q_func_factory: QFunctionFactory
    _gamma: float
    _n_critics: int
    _q_func: Optional[EnsembleDiscreteQFunction]
    _targ_q_func: Optional[EnsembleDiscreteQFunction]
    _optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        device: str,
        observation_scaler: Optional[ObservationScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            observation_scaler=observation_scaler,
            action_scaler=None,
            reward_scaler=reward_scaler,
            device=device,
        )
        self._learning_rate = learning_rate
        self._optim_factory = optim_factory
        self._encoder_factory = encoder_factory
        self._q_func_factory = q_func_factory
        self._gamma = gamma
        self._n_critics = n_critics

        # initialized in build
        self._q_func = None
        self._targ_q_func = None
        self._optim = None

    def build(self) -> None:
        # setup torch models
        self._build_network()

        # setup target network
        self._targ_q_func = copy.deepcopy(self._q_func)

        to_device(self, self._device)

        # setup optimizer after the parameters move to GPU
        self._build_optim()

    def _build_network(self) -> None:
        self._q_func = create_discrete_q_function(
            self._observation_shape,
            self._action_size,
            self._encoder_factory,
            self._q_func_factory,
            n_ensembles=self._n_critics,
        )

    def _build_optim(self) -> None:
        assert self._q_func is not None
        self._optim = self._optim_factory.create(
            self._q_func.parameters(), lr=self._learning_rate
        )

    @train_api
    @torch_api(observation_scaler_targets=["obs_t", "obs_tpn"])
    def update(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._optim is not None

        self._optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_loss(batch, q_tpn)

        loss.backward()
        self._optim.step()

        return loss.cpu().detach().numpy()

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            next_actions = self._targ_q_func(batch.next_observations)
            max_action = next_actions.argmax(dim=1)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
            )

    def _predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func(x).argmax(dim=1)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._predict_best_action(x)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)

    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        assert self._optim
        return self._optim


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._predict_best_action(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
