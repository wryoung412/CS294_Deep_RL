import datetime
import numpy as np
import inspect, os
import tensorflow as tf
from abc import ABC, abstractmethod

class BasePolicy(ABC):
    def __init__(self, env_name):
        # __file__
        policy_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        model_dir = os.path.join(policy_dir, self.type(), env_name)
        model_meta = os.path.join(model_dir, env_name + '.meta')
        print(model_meta)
        assert tf.gfile.Exists(model_meta)
        self.graph = tf.Graph()
        self.sess = tf.Session(graph = self.graph)
        with self.graph.as_default():
            saver = tf.train.import_meta_graph(model_meta)
            saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

    @abstractmethod
    def type():
        pass

    def act(self, ob):
        input = np.reshape(ob, [-1] + list(ob.shape))
        obs_ph = self.graph.get_operation_by_name('obs_ph').outputs[0]
        act_out = self.graph.get_operation_by_name('act_out').outputs[0]
        output = self.sess.run(act_out, feed_dict = {obs_ph: input})
        return output[0]

if __name__ == '__main__':
    print('base policy')
