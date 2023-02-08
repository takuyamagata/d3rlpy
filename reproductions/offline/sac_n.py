import argparse
import d3rlpy
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='maze2d-medium-v1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_critics', type=int, default=10) # original 2
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    
    # dataset, env = d3rlpy.datasets.get_dataset(args.dataset)
    dataset, env = d3rlpy.datasets.get_d4rl(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
    
    sac_n = d3rlpy.algos.SAC(batch_size=256,
                           actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           temp_learning_rate=3e-4,
                           n_steps = 10, # for QDT
                           gamma = 0.999,# for QDT
                           tau = 0.005,  # for QDT
                           actor_encoder_factory=encoder,
                           critic_encoder_factory=encoder,
                           n_critics=args.n_critics,
                           use_gpu=args.gpu)
    
    sac_n.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"SAC_N_{args.dataset}_{args.seed}")
    
if __name__ == '__main__':
    main()
