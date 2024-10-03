import argparse
import d3rlpy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    parser = argparse.ArgumentParser()
	# parser.add_argument('--dataset', type=str, default='halfcheetah-medium-replay-v2')
    parser.add_argument('--dataset', type=str, default='maze2d-large-v1')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--mode', type=str, default='normal') # 'delayed' or 'normal'
    parser.add_argument('--expectile', type=float, default=0.7) 
    parser.add_argument('--weight_temp', type=float, default=3.0) 
    parser.add_argument('--reward_scale', type=float, default=1000)
    parser.add_argument('--n_trials', type=int, default=10)
    args = parser.parse_args()

    #dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset, mode=args.mode)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    
    if args.mode == "delayed":
        print("Delayed reward mode enabled")

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    if args.reward_scale < 0.0:
        reward_scaler = d3rlpy.preprocessing.StandardRewardScaler(mean=1.0, std=1.0) # reward - 1.0
    else:    
        reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(multiplier=args.reward_scale)

    ted_iql = d3rlpy.algos.TedIQL(actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           batch_size=256,
                           weight_temp=args.weight_temp,
                           max_weight=100.0,
                           expectile=args.expectile,
                           gamma_base = 0.99,
                           gamma=0.999,
                           reward_scaler=reward_scaler,
                           use_gpu=args.gpu)

    # workaround for learning scheduler
    ted_iql.create_impl(dataset.get_observation_shape(), dataset.get_action_size())
    scheduler = CosineAnnealingLR(ted_iql.impl._actor_optim, 500000)

    def callback(algo, epoch, total_step):
        scheduler.step()

    ted_iql.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=2500,
            save_interval=10,
            callback=callback,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env, n_trials=args.n_trials),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"TedIQL_{args.dataset}_{args.mode}_exp{args.expectile}_rws{args.reward_scale}_seed{args.seed}",
            with_timestamp=False,)


if __name__ == '__main__':
    main()
