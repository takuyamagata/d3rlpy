import argparse
import d3rlpy
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='walker2d-medium-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_critics', type=int, default=10)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--mode', type=str, default='normal')
    args = parser.parse_args()
    
    #dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
   
    if args.mode == "delayed":
        print("Delayed reward mode enabled")
        for path in dataset.episodes:
            total_return = np.sum(path.rewards)
            for n in range(len(path.rewards)):
                path.rewards[n] = 0
            path.rewards[n] = total_return

    edac = d3rlpy.algos.EDAC(batch_size=256,
                           actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           temp_learning_rate=3e-4,
                           actor_encoder_factory=encoder,
                           critic_encoder_factory=encoder,
                           n_critics=args.n_critics,
                           eta=args.eta,
                           use_gpu=args.gpu)
    
    edac.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"EDAC_{args.dataset}_eta{args.eta}_{args.seed}_{args.mode}",
            with_timestamp=False,)
    
if __name__ == '__main__':
    main()
