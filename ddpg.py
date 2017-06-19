""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import tflearn
import Communicator
import random
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer

# ==========================
#   Training Parameters
# ==========================
# Max training steps
MAX_EPISODES = 50000
# Max episode length
MAX_EP_STEPS = 1000
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.8
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
RENDER_ENV = True
# Use Gym Monitor
GYM_MONITOR_EN = True
# Gym environment
ENV_NAME = 'Pendulum-v0'
# Directory for storing gym results
MONITOR_DIR = './results/gym_ddpg'
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 12024
# Size of replay buffer
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================


class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, (None,)+self.a_dim)

        # Combine the gradients here
        self.actor_gradients = tf.gradients(
            self.out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=(None,)+self.s_dim)
        net = tf.contrib.layers.conv2d(inputs, 32, 1, 1, 
                                     activation_fn=tf.nn.relu,
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                     biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        net = tf.contrib.layers.conv2d(net, 32, 1, 1, 
                                     activation_fn=tf.nn.relu,
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                     biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        net = tf.contrib.layers.conv2d(net, 32, 1, 1, 
                                     activation_fn=tf.nn.relu,
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                     biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        net = tf.contrib.layers.flatten(net)
                                     #kernel_regularizer=tf.nn.l2_loss,
                                     #bias_regularizer=tf.nn.l2_loss,)
        net = tflearn.fully_connected(net, 200, activation='relu')
        net = tflearn.fully_connected(net, 200, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        out = tf.contrib.layers.fully_connected(
            net, sum(self.a_dim), activation_fn=tf.nn.tanh, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
            biases_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        return inputs, out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=(None,) + self.s_dim)
        action = tflearn.input_data(shape=(None,) + self.a_dim)

        net = tf.contrib.layers.conv2d(inputs, 32,1,1, 
                                     activation_fn=tf.nn.relu,
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                     biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        net = tf.contrib.layers.conv2d(net, 32,1, 1, 
                                     activation_fn=tf.nn.relu,
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                     biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        net = tf.contrib.layers.conv2d(net, 32,1, 1, 
                                     activation_fn=tf.nn.relu,
                                     weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                     biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        net = tf.contrib.layers.flatten(net)
                                     #kernel_regularizer=tf.nn.l2_loss,
                                     #bias_regularizer=tf.nn.l2_loss,)
        net = tflearn.fully_connected(net, 300, activation='relu')
        net = tflearn.fully_connected(net, 512, activation='relu')
        net2 = tflearn.fully_connected(action, 512, activation='relu')
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        #out = tflearn.fully_connected(net + net2, 1, activation='tanh')
        #out = tflearn.fully_connected(net + net2, 1, activation=None)
        #out = tflearn.fully_connected(net2, 1, activation=None)
        out = tf.contrib.layers.fully_connected(
            net+net2, sum(self.a_dim), activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
            biases_initializer=tf.contrib.layers.xavier_initializer(uniform=True))

        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================


def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================


def train(sess, world, actor, critic, state_dim):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    for i in range(MAX_EPISODES):

        Communicator.init_world(world)

        ep_reward = 0
        ep_ave_max_q = 0

        s, s2 = None, None
        _s, _last_s = None, None
        for j in range(MAX_EP_STEPS):
            _s = np.dot(Communicator.get_screen_pixels(world, state_dim[0], state_dim[1]), [0.299, 0.587, 0.114]).astype(np.float32)/256.

            if(_last_s!=None):
                s = np.concatenate((_last_s[...,np.newaxis], _s[...,np.newaxis]), axis=2)
            else:
                s = np.concatenate((_s[...,np.newaxis], _s[..., np.newaxis]), axis=2)
            _last_s = _s

            # Added exploration noise
            a = actor.predict(np.reshape(s, (1,)+s.shape))
            a[0][0] += (random.random()*4-2) * (100. / (100. + i))
            #a[0][1] += (random.random()*2-1) * (1. / (1. + i))
            #a = actor.predict(np.reshape(s, (1,)+s.shape))
              
            #Communicator.send_action(world, "move_forward", 1)
            Communicator.send_action(world, "absolute_move", a[0][0],0,0)

            _s2 = np.dot(Communicator.get_screen_pixels(world, state_dim[0], state_dim[1]), [0.299, 0.587, 0.114]).astype(np.float32)/256.
            s2 = np.concatenate((_s[...,np.newaxis], _s2[...,np.newaxis]), axis=2)
            r = Communicator.get_reward(world)
            terminal = 0
            if world.robot.x > world.w or world.robot.x < 0 or world.robot.y > world.h or world.robot.y < 0:
                terminal = 1
                #Communicator.init_world(world)

            if RENDER_ENV:
                Communicator.show(world, 300, 300)
            replay_buffer.add(np.reshape(s, actor.s_dim), np.reshape(a, actor.a_dim), r,
                              terminal, np.reshape(s2, actor.s_dim))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal: break


        summary_str = sess.run(summary_ops, feed_dict={
            summary_vars[0]: ep_reward,
            summary_vars[1]: ep_ave_max_q / float(j)
        })

        writer.add_summary(summary_str, i)
        writer.flush()

        print('| Reward: %.2i' % int(ep_reward), " | Episode", i, \
            '| Qmax: %.4f' % (ep_ave_max_q / float(j)))


def main(_):
    with tf.Session() as sess:

        world = Communicator.gen_world(1500, 1500)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        state_dim = (32, 32, 2)
        action_dim = (1,)

        actor = ActorNetwork(sess, state_dim, action_dim, 
                             ACTOR_LEARNING_RATE, TAU)

        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        train(sess, world, actor, critic, state_dim)

if __name__ == '__main__':
    tf.app.run()
