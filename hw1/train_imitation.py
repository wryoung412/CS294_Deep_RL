#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python train_imitation.py RoboschoolAnt-v1 --num_rollouts 20 --epochs 100 --max_batches 10000
"""


import argparse
import pickle
import os, glob
import tensorflow as tf
import numpy as np
import gym, roboschool
import importlib
from OpenGL import GLU
from sklearn.utils import shuffle
import time

import tensorflow.contrib.slim as slim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--max_batches', type=int, default=10000,
                        help='Max number of batches to train')
    args = parser.parse_args()

    print('training immitation...')

    data_path = os.path.join('data', args.env)
    file_name = 'train_{}.p'.format(args.num_rollouts)
    with open(os.path.join(data_path, file_name), 'rb') as f:
        expert_data = pickle.load(f)

    X = expert_data['observations']
    Y = expert_data['actions']
    assert len(Y.shape) == 2
    print('number of training data:', len(X))

    # for debug purpose
    # TODO: ask Alex about selu at output?
    env = gym.make(args.env)
    print(env.action_space.low, env.action_space.high)

    obs_ph = tf.placeholder(tf.float32, shape = [None] + list(X.shape[1:]), name = 'obs_ph')
    act_ph = tf.placeholder(tf.float32, shape = [None] + list(Y.shape[1:]), name = 'act_ph')
    hid = slim.fully_connected(obs_ph, 32, activation_fn=tf.nn.relu)
    hid = slim.fully_connected(hid, 32, activation_fn=tf.nn.relu)
    # dummy
    hid = slim.fully_connected(hid, Y.shape[1], activation_fn=None)
    act_out = tf.identity(hid, name = 'act_out')
    
    loss = tf.losses.mean_squared_error(act_out, act_ph)
    tf.summary.scalar('losses/total_loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_step = optimizer.minimize(loss)
    
    batch_size = 100
    batch_index = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf_board = os.path.join('/tmp/gube/imitation', args.env)
        # [print(f) for f in glob.glob(os.path.join(tf_board, '*'))]
        # [os.remove(f) for f in glob.glob(os.path.join(tf_board, '*'))]
        writer = tf.summary.FileWriter(os.path.join(tf_board, str(int(time.time()))))
        writer.add_graph(sess.graph)
        tf.summary.scalar('loss', loss)
        merged_summary = tf.summary.merge_all()

        for i in range(args.epochs):
            x, y = shuffle(X, Y)
            print('Epoch {}'.format(i + 1))
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
            print('Batch {}'.format(batch_index))
            if batch_index >= args.max_batches:
                break

        saver = tf.train.Saver()
        saver.save(sess, os.path.join('policy/imitation', args.env, args.env))



if __name__ == '__main__':
    main()

    
