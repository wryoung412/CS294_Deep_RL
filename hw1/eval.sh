#!/usr/bin/env bash
set -e

env=RoboschoolHumanoid-v1
max_batches=10000

# each rollout is about 10 batches
# num of batches = 20 * 10 * 50 = 10000 batches
python train_imitation.py $env --num_rollouts 20 --epochs 50 --max_batches ${max_batches}
# there are 10 rollouts on average
# num of batches = 20 iterations * 10 rollouts on average  * 10 batches per rollout * 5 epochs per rollout = 10000 batches
python train_dagger.py $env --num_rollouts 20 --num_experts 1 --epochs_per_rollout 5 --max_batches ${max_batches}

num_test_rollouts=20
python policy_eval.py imitation --env $env --num_rollouts ${num_test_rollouts}
python policy_eval.py dagger --env $env --num_rollouts ${num_test_rollouts}
