from typing import Any, Dict, Optional, Sequence

from ..argument_utility import (
    ActionScalerArg,
    EncoderArg,
    RewardScalerArg,
    ScalerArg,
    UseGPUArg,
    check_encoder,
    check_use_gpu,
)
from ..constants import IMPL_NOT_INITIALIZED_ERROR, ActionSpace
from ..dataset import TransitionMiniBatch
from ..gpu import Device
from ..models.encoders import EncoderFactory
from ..models.optimizers import AdamFactory, OptimizerFactory
from .base import AlgoBase
from .torch.ted_iql_impl import TedIQLImpl


class TedIQL(AlgoBase):
    r"""Implicit Q-Learning algorithm.

    IQL is the offline RL algorithm that avoids ever querying values of unseen
    actions while still being able to perform multi-step dynamic programming
    updates.

    There are three functions to train in IQL. First the state-value function
    is trained via expectile regression.

    .. math::

        L_V(\psi) = \mathbb{E}_{(s, a) \sim D}
            [L_2^\tau (Q_\theta (s, a) - V_\psi (s))]

    where :math:`L_2^\tau (u) = |\tau - \mathbb{1}(u < 0)|u^2`.

    The Q-function is trained with the state-value function to avoid query the
    actions.

    .. math::

        L_Q(\theta) = \mathbb{E}_{(s, a, r, a') \sim D}
            [(r + \gamma V_\psi(s') - Q_\theta(s, a))^2]

    Finally, the policy function is trained by using advantage weighted
    regression.

    .. math::

        L_\pi (\phi) = \mathbb{E}_{(s, a) \sim D}
            [\exp(\beta (Q_\theta - V_\psi(s))) \log \pi_\phi(a|s)]

    References:
        * `Kostrikov et al., Offline Reinforcement Learning with Implicit
          Q-Learning. <https://arxiv.org/abs/2110.06169>`_

    Args:
        actor_learning_rate (float): learning rate for policy function.
        critic_learning_rate (float): learning rate for Q functions.
        actor_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the actor.
        critic_optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            optimizer factory for the critic.
        actor_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the actor.
        critic_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the critic.
        value_encoder_factory (d3rlpy.models.encoders.EncoderFactory or str):
            encoder factory for the value function.
        batch_size (int): mini-batch size.
        n_frames (int): the number of frames to stack for image observation.
        n_steps (int): N-step TD calculation.
        gamma (float): discount factor.
        tau (float): target network synchronization coefficiency.
        n_critics (int): the number of Q functions for ensemble.
        expectile (float): the expectile value for value function training.
        weight_temp (float): inverse temperature value represented as
            :math:`\beta`.
        max_weight (float): the maximum advantage weight value to clip.
        use_gpu (bool, int or d3rlpy.gpu.Device):
            flag to use GPU, device ID or device.
        scaler (d3rlpy.preprocessing.Scaler or str): preprocessor.
            The available options are `['pixel', 'min_max', 'standard']`.
        action_scaler (d3rlpy.preprocessing.ActionScaler or str):
            action preprocessor. The available options are ``['min_max']``.
        reward_scaler (d3rlpy.preprocessing.RewardScaler or str):
            reward preprocessor. The available options are
            ``['clip', 'min_max', 'standard']``.
        impl (d3rlpy.algos.torch.iql_impl.IQLImpl): algorithm implementation.

    """

    _actor_learning_rate: float
    _critic_learning_rate: float
    _actor_optim_factory: OptimizerFactory
    _critic_optim_factory: OptimizerFactory
    _actor_encoder_factory: EncoderFactory
    _critic_encoder_factory: EncoderFactory
    _value_encoder_factory: EncoderFactory
    _tau: float
    _taylor_order: int
    _expectile: float
    _weight_temp: float
    _max_weight: float
    _use_gpu: Optional[Device]
    _impl: Optional[TedIQLImpl]
    _disable_critic_update: bool
    _disable_actor_update: bool
    _rtg_in_r: bool

    def __init__(
        self,
        *,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        actor_optim_factory: OptimizerFactory = AdamFactory(),
        critic_optim_factory: OptimizerFactory = AdamFactory(),
        actor_encoder_factory: EncoderArg = "default",
        critic_encoder_factory: EncoderArg = "default",
        value_encoder_factory: EncoderArg = "default",
        batch_size: int = 256,
        n_frames: int = 1,
        n_steps: int = 1,
        gamma_base: float = 0.99,
        gamma: float = 0.999,
        tau: float = 0.005,
        n_critics: int = 2,
        taylor_order: int = 4,
        expectile: float = 0.7,
        weight_temp: float = 3.0,
        max_weight: float = 100.0,
        use_gpu: UseGPUArg = False,
        scaler: ScalerArg = None,
        action_scaler: ActionScalerArg = None,
        reward_scaler: RewardScalerArg = None,
        disable_critic_update = False,
        disable_actor_update = False,
        rtg_in_r = False,
        impl: Optional[TedIQLImpl] = None,
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            n_frames=n_frames,
            n_steps=n_steps,
            gamma=gamma,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            kwargs=kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = check_encoder(actor_encoder_factory)
        self._critic_encoder_factory = check_encoder(critic_encoder_factory)
        self._value_encoder_factory = check_encoder(value_encoder_factory)
        self._tau = tau
        self._n_critics = n_critics
        self._gamma_base = gamma_base
        self._taylor_order = taylor_order
        self._expectile = expectile
        self._weight_temp = weight_temp
        self._max_weight = max_weight
        self._use_gpu = check_use_gpu(use_gpu)
        self._disable_critic_update = disable_critic_update
        self._disable_actor_update = disable_actor_update
        self._rtg_in_r = rtg_in_r
        self._impl = impl

    def _create_impl(
        self, observation_shape: Sequence[int], action_size: int
    ) -> None:
        self._impl = TedIQLImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=self._actor_learning_rate,
            critic_learning_rate=self._critic_learning_rate,
            actor_optim_factory=self._actor_optim_factory,
            critic_optim_factory=self._critic_optim_factory,
            actor_encoder_factory=self._actor_encoder_factory,
            critic_encoder_factory=self._critic_encoder_factory,
            value_encoder_factory=self._value_encoder_factory,
            gamma_base=self._gamma_base,
            gamma=self._gamma,
            tau=self._tau,
            n_critics=self._n_critics,
            taylor_order=self._taylor_order,
            expectile=self._expectile,
            weight_temp=self._weight_temp,
            max_weight=self._max_weight,
            use_gpu=self._use_gpu,
            scaler=self._scaler,
            action_scaler=self._action_scaler,
            reward_scaler=self._reward_scaler,
            rtg_in_r=self._rtg_in_r
        )
        self._impl.build()

    def _update(self, batch: TransitionMiniBatch) -> Dict[str, float]:
        assert self._impl is not None, IMPL_NOT_INITIALIZED_ERROR

        metrics = {}

        if not self._disable_critic_update:
            critic_loss, value_loss = self._impl.update_critic(batch)
            metrics.update({"critic_loss": critic_loss, "value_loss": value_loss})

        if not self._disable_actor_update:
            actor_loss = self._impl.update_actor(batch)
            metrics.update({"actor_loss": actor_loss})

        if not self._disable_critic_update:
            self._impl.update_critic_target()

        return metrics

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS
