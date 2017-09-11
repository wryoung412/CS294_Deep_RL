#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python train_dagger.py RoboschoolAnt-v1 --num_rollouts 20 --num_experts 1 --epochs_per_rollout 5 --max_batches 10000
"""

import time
import argparse
import pickle
import os, glob
import tensorflow as tf
import numpy as np
import gym, roboschool
import importlib
from OpenGL import GLU
from sklearn.utils import shuffle

import tensorflow.contrib.slim as slim

# np.vstack((A, a))
# def AppendRow(A, a):
#     return np.append(A, a[None, :], axis = 0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of roll outs')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='Max number of steps per roll out')
    parser.add_argument('--num_experts', type=int, default=1,
                        help='Number of expert roll outs')
    parser.add_argument('--epochs_per_rollout', type=int, default=5,
                        help='Number of epochs to train for every roll out')
    parser.add_argument('--max_batches', type=int, default=10000,
                        help='Max number of batches to train')
    args = parser.parse_args()

    print('training DAgger...')

    # set up
    env = gym.make(args.env)
    obs_shape = list(env.observation_space.shape)
    act_shape = list(env.action_space.shape)
    assert len(act_shape) == 1
    print(env.action_space.low, env.action_space.high)
    max_steps = args.max_steps or env.spec.timestep_limit
    policy_module = importlib.import_module("policy.expert")
    expert = policy_module.get_policy(args.env)

    obs_ph = tf.placeholder(tf.float32, shape = [None] + obs_shape, name = 'obs_ph')
    act_ph = tf.placeholder(tf.float32, shape = [None] + act_shape, name = 'act_ph')
    hid = slim.fully_connected(obs_ph, 32, activation_fn=tf.nn.relu)
    hid = slim.fully_connected(hid, 32, activation_fn=tf.nn.relu)
    # dummy
    hid = slim.fully_connected(hid, act_shape[0], activation_fn=None)
    act_out = tf.identity(hid, name = 'act_out')
    
    loss = tf.losses.mean_squared_error(act_out, act_ph)
    tf.summary.scalar('losses/total_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_step = optimizer.minimize(loss)
    
    batch_size = 100
    batch_index = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf_board = os.path.join('/tmp/gube/dagger', args.env)
        # [os.remove(f) for f in glob.glob(os.path.join(tf_board, '*'))]
        writer = tf.summary.FileWriter(os.path.join(tf_board, str(int(time.time()))))
        writer.add_graph(sess.graph)
        tf.summary.scalar('loss', loss)
        merged_summary = tf.summary.merge_all()

        X = np.empty([0] + obs_shape) # float
        Y = np.empty([0] + act_shape) # float
        returns = []
        for i in range(args.num_rollouts):
            print('rollout', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # obs[None, :] = [obs], adding 1 to the shape
                policy_action = sess.run(act_out, feed_dict = {obs_ph: obs[None, :]})[0]
                expert_action = expert.act(obs)
                X = np.vstack((X, obs))
                Y = np.vstack((Y, expert_action))
                if i < args.num_experts:
                    obs, r, done, _ = env.step(expert_action)
                else:
                    obs, r, done, _ = env.step(policy_action)
                totalr += r
                steps += 1
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

            for j in range(args.epochs_per_rollout):
                x, y = shuffle(X, Y)
                offset = 0
                while offset + batch_size <= len(x):
                    start, end = offset, offset + batch_size
                    sess.run(train_step,
                             feed_dict = {
                                 obs_ph: x[start:end],
                                 act_ph: y[start:end]
                             })
                    offset = end
                    batch_index += 1
                    if batch_index >= args.max_batches:
                        break
                
            s = sess.run(merged_summary, feed_dict = {obs_ph: x, act_ph: y})
            writer.add_summary(s, batch_index)
            if batch_index >= args.max_batches:
                break

            print('batch', batch_index)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
            
        saver = tf.train.Saver()
        saver.save(sess, os.path.join('policy/dagger', args.env, args.env))


if __name__ == '__main__':
    main()

    
