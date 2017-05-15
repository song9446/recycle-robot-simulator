import tensorflow as tf

from collections import deque, namedtuple


class DDPG:
    def __init__(state_dim, batch_size, num_actions, summary_dir=None):
        # summary writer
        #if summary_dir: summary_writer = tf.train.SummaryWriter(summary_dir)
        #else: summary_writer = 
        # process the input
        # q-value estimater neural network
        with tf.variable_scope("actor_network"):
            state = tf.placeholder(shape=(batch_size,)+state_dim, dtype=tf.uint8, name="state")
            state = tf.to_float(state)
            conv = tf.contrib.layers.conv2d(
                state, 32, 8, 4, activation_fn=tf.nn.relu)
            conv = tf.contrib.layers.conv2d(
                conv, 64, 4, 2, activation_fn=tf.nn.relu)
            conv = tf.contrib.layers.conv2d(
                conv, 64, 3, 1, activation_fn=tf.nn.relu)
            flattened = tf.contrib.layers.flatten(conv)
            fc = tf.contrib.layers.fully_connected(flattend, 512)
            predict_action = tf.contrib.layers.fully_connected(fc, num_actions, activation_fn=tf.nn.tanh)
            
        with tf.variable_scope("critic_network"):
            action = tf.placeholder(shape=(batch_size, num_actions), dtype=tf.float32, name="action")
            conv = tf.contrib.layers.conv2d(
                state, 32, 8, 4, activation_fn=tf.nn.relu)
            conv = tf.contrib.layers.conv2d(
                conv, 64, 4, 2, activation_fn=tf.nn.relu)
            conv = tf.contrib.layers.conv2d(
                conv, 64, 3, 1, activation_fn=tf.nn.relu) flattened = tf.contrib.layers.flatten(conv)
            fc = tf.contrib.layers.fully_connected(flattend, 512)
            predict_reward = tf.contrib.layers.fully_connected(fc, 1)
        
    def train():
    def test():


def process_state(state, w, h):
    
