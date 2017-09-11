#!/usr/bin/env bash
set -e

env=RoboschoolAnt-v1
rm -rf /tmp/gube/dagger/$env /tmp/gube/imitation/$env

num_rollout=100
max_batches=20000


# Ant:
# 100 samples per batch. Each rollout is about 10 batches
# Imitation: --num_rollouts 20 --epochs 50
# num of batches = 20 rollouts * 10 batches per rollout * 50 epochs = 10000
# DAgger: --num_rollouts 20 --num_experts 1 --epochs_per_rollout 5
# there are 10 rollouts on average
# num of batches = 20 iterations * 10 rollouts on average  * 10 batches per rollout * 5 epochs per rollout = 10000

python train_imitation.py $env --num_rollouts ${num_rollout} --epochs 50 --max_batches ${max_batches}
python train_dagger.py $env --num_rollouts ${num_rollout} --num_experts 1 --epochs_per_rollout 5 --max_batches ${max_batches}

num_test_rollouts=20
python policy_eval.py expert --env $env --num_rollouts ${num_test_rollouts}
python policy_eval.py imitation --env $env --num_rollouts ${num_test_rollouts}
python policy_eval.py dagger --env $env --num_rollouts ${num_test_rollouts}
