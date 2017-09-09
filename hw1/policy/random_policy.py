from __future__ import absolute_import

import random
import datetime
import numpy as np
import gym, roboschool

def get_policy(env_name):
    return RandomPolicy(env_name)

class RandomPolicy:
    def __init__(self, env_name):
        env = gym.make(env_name)
        self.action_space = env.action_space
        random.seed(datetime.datetime.now().timestamp())

    def act(self, ob):
        low = self.action_space.low
        high = self.action_space.high
        shape = self.action_space.shape
        return np.random.uniform(low, high, shape)

    @staticmethod
    def env():
        return ''
