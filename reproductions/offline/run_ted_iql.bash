#/bin/bash
dataset="maze2d-large-v1"

n_trials=100
expectile=0.7
weight_temp=3.0
reward_scale=1000
for seed in 111 222 # 123 231 312
do
    python ted_iql.py --dataset $dataset --expectile $expectile --weight_temp $weight_temp --reward_scale $reward_scale --n_trials $n_trials
done

n_trials=100
expectile=0.9
weight_temp=10.0
reward_scale=1000
for seed in 111 222 # 123 231 312
do
    python ted_iql.py --dataset $dataset --expectile $expectile --weight_temp $weight_temp --reward_scale $reward_scale --n_trials $n_trials
done
