#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python gen_expert.py RoboschoolHumanoid-v1 --num_rollouts 20
"""

import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import importlib
from OpenGL import GLU

import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading expert policy')
    module_name = 'policy.experts.{}'.format(args.env)
    policy_module = importlib.import_module(module_name)
    print('loaded')

    env, policy = policy_module.get_env_and_policy()
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy.act(obs)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                    'actions': np.array(actions)}
    data_path = os.path.join('data', args.env)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_name = 'train_{}.p'.format(args.num_rollouts)
    with open(os.path.join(data_path, file_name), 'wb') as f:
        pickle.dump(expert_data, f)

if __name__ == '__main__':
    main()
