import tensorflow as tf
import tensorflow.contrib.slim as slim

class SimpleNet:
    # Could set the parameter here. More flexible than a function. 
    def __init__(self):
        pass

    def CreateGraph(self, obs_ph, act_ph):
        act_shape = act_ph.get_shape().as_list()
        assert len(act_shape) == 2
        
        hid = slim.fully_connected(obs_ph, 32, activation_fn=tf.nn.relu)
        hid = slim.fully_connected(hid, 32, activation_fn=tf.nn.relu)
        # dummy
        hid = slim.fully_connected(hid, act_shape[1], activation_fn=None)
        act_out = tf.identity(hid, name = 'act_out')
        
        loss = tf.losses.mean_squared_error(act_out, act_ph)
        tf.summary.scalar('losses/total_loss', loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        train_step = optimizer.minimize(loss)

        return act_out, loss, train_step
