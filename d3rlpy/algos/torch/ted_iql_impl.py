from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ...gpu import Device
from ...models.builders import (
    create_non_squashed_normal_policy,
    create_value_function,
    create_continuous_q_function,
)
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import MeanQFunctionFactory
from ...models.torch import NonSquashedNormalPolicy, ValueFunction
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .ddpg_impl import DDPGBaseImpl


class TedIQLImpl(DDPGBaseImpl):
    _policy: Optional[NonSquashedNormalPolicy]
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _value_encoder_factory: EncoderFactory
    _value_func: Optional[ValueFunction]
    _rtg_in_r: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        value_encoder_factory: EncoderFactory,
        gamma_base: float,
        gamma: float,
        tau: float,
        n_critics: int,
        taylor_order: int,
        expectile: float,
        weight_temp: float,
        max_weight: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        rtg_in_r: bool,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=MeanQFunctionFactory(),
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._value_encoder_factory = value_encoder_factory
        self._value_func = None
        self._gamma_base = gamma_base
        self._taylor_order = taylor_order
        self._rtg_in_r = rtg_in_r

    def _build_actor(self) -> None:
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )

    def _build_critic(self) -> None:
        q_funcs = []
        v_funcs = []
        for _ in range(self._taylor_order+1):
            q_funcs.append(
                create_continuous_q_function(
                    self._observation_shape,
                    self._action_size,
                    self._critic_encoder_factory,
                    self._q_func_factory,
                    n_ensembles=self._n_critics,
                )
            )
            v_funcs.append(
                create_value_function(
                    self._observation_shape, self._value_encoder_factory,
                )
            )   
        self._q_func = nn.ModuleList(q_funcs)
        self._value_func = nn.ModuleList(v_funcs)
            

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._critic_learning_rate
        )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        with torch.no_grad():
            rewards = [batch.rewards]
            for v_func in self._value_func[:-1]:
                rewards.append(
                    v_func(batch.next_observations) * (self._gamma - self._gamma_base)
                )
            rewards = torch.stack(rewards, dim=0)
        
        td_sum = torch.tensor(
            0.0, dtype=torch.float32, device=batch.observations.device
        )
        for n, q_func in enumerate(self._q_func):
            loss = q_func.compute_error(
                    observations=batch.observations,
                    actions=batch.actions,
                    rewards=rewards[n],
                    target=q_tpn[n],
                    terminals=batch.terminals,
                    gamma=self._gamma_base**batch.n_steps,
            )
            td_sum += loss.mean()
        return td_sum

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        with torch.no_grad():
            target = []
            for v_func in self._value_func:
                target.append(
                    v_func(batch.next_observations)
                )
            return torch.stack(target, dim=0)

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy

        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch)

        return -(weight * log_probs).mean()

    def _compute_weight(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = []
        for targ_q_func in self._targ_q_func:
            if self._rtg_in_r:
                q_t.append(
                    batch.rewards # relabelled RTGs stored in rewards in batch
                )
            else:
                q_t.append(
                    targ_q_func(batch.observations, batch.actions, "min")   
                )
        q_t = torch.stack(q_t, dim=0)
        v_t = []
        for v_func in self._value_func:
            v_t.append(
                v_func(batch.observations)
            )
        v_t = torch.stack(v_t, dim=0)
        adv = torch.sum(q_t - v_t, dim=0)
        return (self._weight_temp * adv).exp().clamp(max=self._max_weight)

    def compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = []
        for targ_q_func in self._targ_q_func:
            q_t.append(
                targ_q_func(batch.observations, batch.actions, "min")
            )
        q_t = torch.stack(q_t, dim=0)
        v_t = []
        for v_func in self._value_func:
            v_t.append(
                v_func(batch.observations)
            )
        v_t = torch.stack(v_t, dim=0)
        # v_t = self._value_func(batch.observations)
        diff = q_t.detach() - v_t
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        return (weight * (diff**2)).mean()

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        # compute Q-function loss
        q_tpn = self.compute_target(batch)
        q_loss = self.compute_critic_loss(batch, q_tpn)

        # compute value function loss
        v_loss = self.compute_value_loss(batch)

        loss = q_loss + v_loss

        loss.backward()
        self._critic_optim.step()

        return q_loss.cpu().detach().numpy(), v_loss.cpu().detach().numpy()
