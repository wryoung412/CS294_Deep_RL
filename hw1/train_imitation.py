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

import models


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
    obs_shape = list(env.observation_space.shape)
    act_shape = list(env.action_space.shape)
    print(env.action_space.low, env.action_space.high)

    obs_ph = tf.placeholder(tf.float32, shape = [None] + obs_shape, name = 'obs_ph')
    act_ph = tf.placeholder(tf.float32, shape = [None] + act_shape, name = 'act_ph')
    net = models.SimpleNet()
    act_out, loss, train_step = net.CreateGraph(obs_ph, act_ph)
    tf.summary.scalar('losses/total_loss', loss)
    
    batch_size = 100
    batch_index = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf_board = os.path.join('/tmp/gube/imitation', args.env)
        # [print(f) for f in glob.glob(os.path.join(tf_board, '*'))]
        # [os.remove(f) for f in glob.glob(os.path.join(tf_board, '*'))]
        writer = tf.summary.FileWriter(os.path.join(tf_board, str(int(time.time()))))
        writer.add_graph(sess.graph)
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

    
