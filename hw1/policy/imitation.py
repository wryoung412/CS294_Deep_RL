import random
import datetime
import numpy as np
import inspect, os
import tensorflow as tf
from .base import BasePolicy

def get_policy(env_name):
    return ImitationPolicy(env_name)

class ImitationPolicy(BasePolicy):
    def type():
        return 'imitation'
