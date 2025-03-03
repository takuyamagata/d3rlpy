import argparse

from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy
import d4rl.gym_mujoco # not required...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="maze2d-large-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--weight_temp", type=float, default=3.0)
    parser.add_argument("--expectile", type=float, default=0.7)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

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
        expectile=args.expectile,
        weight_temp=args.weight_temp,
        max_weight=100.0,

        reward_scaler=reward_scaler,
    ).create(device=args.gpu)

    seq_iql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=2500,
        save_interval=10,
        callback=callback,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"SeqIQL_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
