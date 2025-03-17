import argparse
from datetime import datetime
from typing import Optional, Union

import gym
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy
from d3rlpy.algos import CQL, IQL, SeqIQL
from d3rlpy.dataset import InfiniteBuffer, ReplayBuffer
from d3rlpy.types import NDArray

import d4rl.gym_mujoco # not required...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v2")
    parser.add_argument("--context_size", type=int, default=20)
    parser.add_argument("--model_file", type=str, default=None)
    parser.add_argument(
        "--q_learning_type",
        type=str,
        default="cql",
        choices=["cql", "iql", "seq_iql", "none"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_action_samples", type=int, default=10)
    parser.add_argument("--delayed_reward", action="store_true")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # create postfix of log directories
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_postfix = f"{env.spec.id}_delayed-reward_{args.seed}_{timestamp}" \
        if args.delayed_reward else f"{env.spec.id}_{args.seed}_{timestamp}"

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # delayed reward
    if args.delayed_reward:
        print("Modified dataset to delayed reward...")
        delayed_reward(dataset._buffer)

    # first fit Q-learning algorithm to the dataset
    if args.model_file is not None:
        # load model and assert type
        q_algo_loaded = d3rlpy.load_learnable(args.model_file)
        if not isinstance(q_algo_loaded, (CQL, IQL)):
            raise ValueError(
                "The loaded model is not an instance of CQL or IQL."
            )
        # cast to the expected type
        q_algo = q_algo_loaded
    else:
        if args.q_learning_type == "cql":
            q_algo = fit_cql(
                dataset=dataset,
                env=env,
                gpu=args.gpu,
                log_postfix=log_postfix,
                compile=args.compile,
            )
        elif args.q_learning_type == "iql":
            q_algo = fit_iql(
                dataset=dataset,
                env=env,
                gpu=args.gpu,
                log_postfix=log_postfix,
                compile=args.compile,
            )
        elif args.q_learning_type == "seq_iql":
            q_algo = fit_seq_iql(
                dataset=dataset,
                env=env,
                gpu=args.gpu,
                log_postfix=log_postfix,
                compile=args.compile,
            )
        elif args.q_learning_type == "none":
            # Skip Q-learning (DT only)
            q_algo = None
        else:
            raise ValueError(f"invalid q_learning_type: {args.q_learning_type}")

    # relabel dataset RTGs with the learned value functions
    if q_algo is None:
        print("Skipping relabeling dataset with RTGs...")
    else:
        print("Relabeling dataset with RTGs...")
        assert isinstance(dataset._buffer, InfiniteBuffer)
        relabel_dataset_rtg(
            buffer=dataset._buffer,
            q_algo=q_algo,
            k=args.context_size,
            num_action_samples=args.num_action_samples,
        )

    # fit decision transformer to the relabeled dataset
    fit_dt(
        dataset=dataset,
        env=env,
        context_size=args.context_size,
        gpu=args.gpu,
        log_prefix=f"QDT-{args.q_learning_type}",
        log_postfix=log_postfix,
        compile=args.compile,
    )


""" --------------------------------------------------------------------
    Augment dataset
-------------------------------------------------------------------- """


def relabel_dataset_rtg(
    buffer: InfiniteBuffer,
    q_algo: Union[CQL, IQL, SeqIQL],
    k: int,
    num_action_samples: int,
) -> None:
    """
    Relabel RTG (reward-to-go) to the given dataset using the given Q-function.

    Args:
        buffer (InfiniteBuffer): Buffer holding trajectory dataset.
        q_algo (Union[CQL, IQL, SeqIQL]): Trained Q-learning algoirthm.
        k (int): Context length for DT.
        num_action_samples (int): The number of action samples for
            V function estimation. Defaults to 10.
    """
    prev_idx = -1
    for n in range(buffer.transition_count):
        episode, idx = buffer._transitions[-n - 1]  # get transitions backwards
        if idx > prev_idx:
            # get values for all observations in the episode
            values = []
            for _ in range(num_action_samples):
                sampled_actions = q_algo.sample_action(episode.observations)
                v = q_algo.predict_value(episode.observations, sampled_actions)
                values.append(
                    v if q_algo.reward_scaler is None 
                          else q_algo.reward_scaler.reverse_transform(v)
                )
            value = np.array(values).mean(axis=0)
            rewards = np.squeeze(episode.rewards, axis=1)
            rtg = 0
        else:
            start = max(0, idx - k + 1)
            rtg = rewards[idx] + np.maximum(rtg, value[idx + 1])  # relabel rtg
            relabelled_rewards = np.zeros_like(rewards)
            relabelled_rewards[idx] = rtg
            relabelled_rewards[start:idx] = rewards[start:idx]
            relabelled_episode = d3rlpy.dataset.components.Episode(
                observations=episode.observations,
                actions=episode.actions,
                rewards=np.expand_dims(relabelled_rewards, axis=1),
                terminated=episode.terminated,
            )
            buffer._transitions[-n - 1] = (relabelled_episode, idx)

        prev_idx = idx

    return

def delayed_reward(buffer: InfiniteBuffer):
    """
    Relabels the rewards in the episodes stored in the buffer such that each episode's 
    final reward is the sum of all rewards in that episode. 
    Args:
        buffer (InfiniteBuffer): The buffer containing episodes and transitions to be relabelled.
    Returns:
        None
    """

    k = 0 # episode index
    for n in range(buffer.transition_count):
        episode, idx = buffer._transitions[n]  # get transitions backwards
        if idx == 0:
            relabelled_rewards = np.zeros_like(episode.rewards)
            relabelled_rewards[episode.transition_count-1] = np.sum(episode.rewards)
            relabelled_episode = d3rlpy.dataset.components.Episode(
                observations=episode.observations,
                actions=episode.actions,
                rewards=relabelled_rewards,
                terminated=episode.terminated,
            )
            buffer._episodes[k] = relabelled_episode
            k += 1
        
        buffer._transitions[n] = (relabelled_episode, idx)

    return

""" --------------------------------------------------------------------
    Fit offline RL algorithms to the given dataset.
-------------------------------------------------------------------- """


def fit_cql(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    gpu: Optional[int],
    log_postfix: str,
    compile: bool,
) -> CQL:
    """
    Fit the CQL algorithm to the given dataset and environment.

    Args:
        dataset (ReplayBuffer): Dataset for the training.
        env (gym.Env): The environment instance.
        gpu (Optional[int]): The GPU device ID..
        log_postfix (str): The postfix of experiment name.
        compile (bool): Flag to enable compilation.

    Return:
        Trained CQL agent.
    """
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    if "medium-v0" in env.spec.id:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_learning_rate=0.0,
        conservative_weight=conservative_weight,
        compile_graph=compile,
    ).create(device=gpu)

    cql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=50,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CQL_{log_postfix}",
        with_timestamp=False,
    )

    return cql


def fit_iql(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    gpu: Optional[int],
    log_postfix: str,
    compile: bool,
) -> IQL:
    """
    Fit the IQL algorithm to the given dataset and environment.

    Args:
        dataset (ReplayBuffer): Dataset for the training.
        env (gym.Env): The environment instance.
        seed (int): The random seed.
        gpu (Optional[int]): The GPU device ID.
        log_postfix (str): The postfix of experiment name.
        compile (bool): Flag to enable compilation.

    Return:
        Trained IQL agent.
    """
    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    iql = d3rlpy.algos.IQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        actor_optim_factory=d3rlpy.optimizers.AdamFactory(
            lr_scheduler_factory=d3rlpy.optimizers.CosineAnnealingLRFactory(
                T_max=500000
            ),
        ),
        batch_size=256,
        gamma=0.99,
        weight_temp=3.0,
        max_weight=100.0,
        expectile=0.7,
        reward_scaler=reward_scaler,
        compile_graph=compile,
    ).create(device=gpu)

    # workaround for learning scheduler
    iql.build_with_dataset(dataset)
    assert iql.impl
    
    iql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={
            "environment": d3rlpy.metrics.EnvironmentEvaluator(env, n_trials=10)
        },
        experiment_name=f"IQL_{log_postfix}",
        with_timestamp=False,
    )

    return iql


def fit_seq_iql(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    gpu: Optional[int],
    log_postfix: str,
    compile: bool,
) -> IQL:
    """
    Fit the SeqIQL algorithm to the given dataset and environment.

    Args:
        dataset (ReplayBuffer): Dataset for the training.
        env (gym.Env): The environment instance.
        seed (int): The random seed.
        gpu (Optional[int]): The GPU device ID.
        log_postfix (str): The postfix of experiment name.
        compile (bool): Flag to enable compilation.

    Return:
        Trained SeqIQL agent.
    """
    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    seq_iql = d3rlpy.algos.SeqIQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        actor_optim_factory=d3rlpy.optimizers.AdamFactory(
            lr_scheduler_factory=d3rlpy.optimizers.CosineAnnealingLRFactory(
                T_max=500000
            ),
        ),
        batch_size=256,
        gamma_base=0.99,
        gamma=0.999,
        expectile=0.7,
        weight_temp=3.0,
        max_weight=100.0,
        reward_scaler=reward_scaler,
        compile_graph=compile,
    ).create(device=gpu)

    seq_iql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"SeqIQL_{log_postfix}",
        with_timestamp=False,
    )

    return seq_iql


def fit_dt(
    dataset: ReplayBuffer,
    env: gym.Env[NDArray, int],
    context_size: int,
    gpu: Optional[int],
    log_prefix: str,
    log_postfix: str,
    compile: bool,
) -> None:
    """
    Fit decisiton transformer to the given dataset and environment.

    Args:
        dataset (MDPdataset): Dataset for the training.
        env (gym.Env): The environment instance.
        context_size (int): The context size of DT.
        gpu (Optional[int]): The GPU device ID.
        log_postfix (str): The postfix of experiment name.
        compile (bool): Flag to enable compilation.
    """
    if "halfcheetah" in env.spec.id:
        target_return = 6000
    elif "hopper" in env.spec.id:
        target_return = 3600
    elif "walker" in env.spec.id:
        target_return = 5000
    elif "maze2d-umaze-dense" in env.spec.id:
        target_return = 500
    elif "maze2d-umaze" in env.spec.id:
        target_return = 500
    elif "maze2d-medium-dense" in env.spec.id:
        target_return = 300
    elif "maze2d-medium" in env.spec.id:
        target_return = 300
    elif "maze2d-large-dense" in env.spec.id:
        target_return = 500
    elif "maze2d-large" in env.spec.id:
        target_return = 500
    else:
        raise ValueError("unsupported dataset")
    
    if "maze2d" in env.spec.id:
        reward_scaler = 0.02
    else:
        reward_scaler = 0.001

    dt = d3rlpy.algos.DecisionTransformerConfig(
        batch_size=64,
        learning_rate=1e-4,
        optim_factory=d3rlpy.optimizers.AdamWFactory(
            weight_decay=1e-4,
            clip_grad_norm=0.25,
            lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
                warmup_steps=10000
            ),
        ),
        encoder_factory=d3rlpy.models.VectorEncoderFactory(
            [128],
            exclude_last_activation=True,
        ),
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.MultiplyRewardScaler(reward_scaler),
        position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
        context_size=context_size,
        num_heads=1,
        num_layers=3,
        max_timestep=1000,
        compile_graph=compile,
    ).create(device=gpu)

    dt.fit(
        dataset,
        n_steps=100000,
        n_steps_per_epoch=1000,
        save_interval=10,
        eval_env=env,
        eval_target_return=target_return,
        experiment_name=f"{log_prefix}_{log_postfix}",
        with_timestamp=False,
    )


if __name__ == "__main__":
    main()
