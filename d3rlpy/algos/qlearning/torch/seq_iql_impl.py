import dataclasses

import torch
#from torch import nn

from ....models.torch import (
    ActionOutput,
    ContinuousEnsembleQFunctionForwarder,
    NormalPolicy,
    ValueFunction,
    build_gaussian_distribution,
)
from ....torch_utility import TorchMiniBatch
from ....types import Shape, TorchObservation
from .ddpg_impl import (
    DDPGBaseActorLoss,
    DDPGBaseCriticLoss,
    DDPGBaseImpl,
    DDPGBaseModules,
)

__all__ = ["SeqIQLImpl", "SeqIQLModules"]


@dataclasses.dataclass(frozen=True)
class SeqIQLModules(DDPGBaseModules):
    policy: NormalPolicy
    v_funcs: ValueFunction #nn.ModuleList


@dataclasses.dataclass(frozen=True)
class SeqIQLCriticLoss(DDPGBaseCriticLoss):
    q_loss: torch.Tensor
    v_loss: torch.Tensor


class SeqIQLImpl(DDPGBaseImpl):
    _modules: SeqIQLModules
    _expectile: float
    _weight_temp: float
    _max_weight: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: SeqIQLModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma_base: float,
        gamma: float,
        taylor_order: int,
        tau: float,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        compiled: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            compiled=compiled,
            device=device,
        )
        self._gamma_base = gamma_base
        self._taylor_order = taylor_order
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> SeqIQLCriticLoss:
        rewards = [batch.rewards]
        for v_func in self._modules.v_funcs[:-1]:
            rewards.append(
                v_func(batch.next_observations) * (self._gamma - self._gamma_base)
            )
        rewards = torch.stack(rewards, dim=0)

        q_loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma_base**batch.intervals,
        )
        v_loss = self.compute_value_loss(batch)
        return SeqIQLCriticLoss(
            critic_loss=q_loss + v_loss,
            q_loss=q_loss,
            v_loss=v_loss,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            target = []
            for v_func in self._modules.v_funcs:
                target.append(
                    v_func(batch.next_observations)
                )
            return torch.stack(target, dim=0)

    def compute_actor_loss(
        self, batch: TorchMiniBatch, action: ActionOutput
    ) -> DDPGBaseActorLoss:
        # compute log probability
        dist = build_gaussian_distribution(action)
        log_probs = dist.log_prob(batch.actions)
        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)
        return DDPGBaseActorLoss(-(weight * log_probs).mean())

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions
        )
        v_t = []
        for v_func in self._modules.v_funcs:
            v_t.append(
                v_func(batch.observations)
            )
        v_t = torch.stack(v_t, dim=0)
        adv = torch.sum(q_t - v_t, dim=0)
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self._targ_q_func_forwarder.compute_expected_q(
            batch.observations, batch.actions
        )
        v_t = []
        for v_func in self._modules.v_funcs:
            v_t.append(
                v_func(batch.observations)
            )
        v_t = torch.stack(v_t, dim=0)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        dist = build_gaussian_distribution(self._modules.policy(x))
        return dist.sample()
