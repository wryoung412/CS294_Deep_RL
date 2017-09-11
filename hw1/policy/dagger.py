import random
import datetime
import numpy as np
import inspect, os
import tensorflow as tf
from .base import BasePolicy

def get_policy(env_name):
    return DaggerPolicy(env_name)

class DaggerPolicy(BasePolicy):
    def type(self):
        return 'dagger'
