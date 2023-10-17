import argparse
import d3rlpy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='walker2d-medium-replay-v2')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--mode', type=str, default='normal')
    args = parser.parse_args()

    #dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset, mode=args.mode)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)
    
    if args.mode == "delayed":
        print("Delayed reward mode enabled")

    _, test_episodes = train_test_split(dataset, test_size=0.2)
        
    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0)

    iql = d3rlpy.algos.IQL(actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           batch_size=256,
                           weight_temp=3.0,
                           max_weight=100.0,
                           expectile=0.7,
                           reward_scaler=reward_scaler,
                           rtg_in_r=True,
                           use_gpu=args.gpu,)

    # workaround for learning scheduler
    iql.create_impl(dataset.get_observation_shape(), dataset.get_action_size())
    scheduler = CosineAnnealingLR(iql.impl._actor_optim, 500000)

    def callback(algo, epoch, total_step):
        scheduler.step()

    # disable actor update
    iql._disable_critic_update = False
    iql._disable_actor_update = True

    iql.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000, #500000, # 30000,
            n_steps_per_epoch=1000,
            save_interval=100,
            callback=callback,
            scorers={
            },
            experiment_name=f"IQL2_critic_{args.dataset}_{args.seed}_{args.mode}",
            with_timestamp=False,)

    # relabelling rewards with RTGs
    gamma = 1.0 #0.99 # discount factor
    r = dataset.rewards
    _R = 0.0 # temporal RTGs
    if args.gpu is None:
        q_t = iql._impl._targ_q_func(torch.tensor(dataset.observations), 
                                    torch.tensor(dataset.actions), "min").cpu().detach().numpy().reshape(-1)
        q_t = iql.reward_scaler.reverse_transform(q_t)
    else:
        q_t = iql._impl._targ_q_func(torch.tensor(dataset.observations).to(f"cuda:{args.gpu}"), 
                                     torch.tensor(dataset.actions).to(f"cuda:{args.gpu}"), "min").cpu().detach().numpy().reshape(-1)
        q_t = iql.reward_scaler.reverse_transform(q_t)
    num_relabel = 0
    for n in np.arange(len(r)-1, -1, -1): # index backwards
        _R = r[n] + gamma * _R
        if dataset.episode_terminals[n]:
            _R = 0.0 # reset RTGs at terminal time-step
        else:
            # relabelling RTGs with learned value function
            if _R < q_t[n]:
                num_relabel = num_relabel + 1
            # _R = np.maximum(_R, q_t[n])
            _R = q_t[n]
        r[n] = _R
    print(f"Relabelling: {num_relabel} / {len(r)} = {num_relabel/len(r)}")
    
    # define Relabelled dataset
    r_dataset = d3rlpy.dataset.MDPDataset(
        observations=dataset.observations,
        actions=dataset.actions,
        rewards=np.array(r, dtype=np.float32),
        terminals=dataset.terminals,
        episode_terminals=dataset.episode_terminals,
    )

    # disable critic update
    iql._disable_critic_update = True
    iql._disable_actor_update = False

    iql.fit(r_dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=1000,
            save_interval=10,
            callback=callback,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"IQL2_actor_{args.dataset}_{args.seed}_{args.mode}",
            with_timestamp=False,)


if __name__ == '__main__':
    main()
