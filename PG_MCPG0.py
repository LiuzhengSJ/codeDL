#!/usr/bin/env python
# coding: utf-8

# In[2]:
#   ********************************************************************
#   Computer software requirements: Anaconda, Python-3.6, tensorflow-1.14

import gym
import os 
import sys
import itertools
import numpy as np
import tensorflow as tf
from collections import defaultdict, namedtuple

import matplotlib
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')


# In[3]:


env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

print("env.action_sapce:", env.action_space.n)
print("env.observation_sapce:", env.observation_space.shape[0])
print("env.observation_space.high:", env.observation_space.high)
print("env.observation_space.low:", env.observation_space.low)


# In[4]:


class PolicyGradient():
    """
    Policy Gradient REinforcement Learning.
    used a 3 layer neural network as the policy network.
    """
    def __init__(self, n_x, n_y,
                learning_rate=0.01, reward_decay=0.95, load_path=None, save_path=None):
        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.reward_decay = reward_decay
        
        self.episode_states, self.episode_actions, self.episode_rewards = [], [], []
        self.cost_history = []
        
        self.__build_network()
        self.sess = tf.Session()
        
        tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def choose_action(self, state):
        """
        choose action base on given state
        """
        # reshape state to (num_features, 1)
        state = state[:, np.newaxis]
        
        # get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict={self.X: state})
        variable_name = [v.name for v in tf.trainable_variables()]

        # return sampled action
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action
    
    def store_transition(self, state, action, reward):
        """
        Store game memory for network training
        """
        self.episode_states.append(state)
        self.episode_rewards.append(reward)
        
        action__ = np.zeros(self.n_y)
        action__[action] = 1
        self.episode_actions.append(action__)
        
    def learn(self):
        """
        Accroding the game memory traing the network
        """
        # discount and normalize episode reward
        disc_norm_ep_reward = self.__disc_and_norm_rewards()
        
        # train on episodes
        self.sess.run(self.train_op, feed_dict={
            self.X: np.vstack(self.episode_states).T,
            self.Y: np.vstack(self.episode_actions).T,
            self.disc_norm_ep_reward: disc_norm_ep_reward,  
        })
        
        # Reset the episode data
        self.episode_states, self.episode_actions, self.episode_rewards  = [], [], []
        
        return disc_norm_ep_reward
        
    def __build_network(self):
        """
        build the natural network
        """
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(self.n_x, None), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(self.n_y, None), name="Y")
            self.disc_norm_ep_reward = tf.placeholder(tf.float32, [None, ], name="actions_value")

        layer1_units = 10
        layer2_units = 10
        layer_output_units = self.n_y
        
        with tf.name_scope("parameter"):
            W1 = self.__weigfht_variable([layer1_units, self.n_x], "W1")
            b1 = self.__bias_bariable([layer1_units, 1], "b1")
            W2 = self.__weigfht_variable([layer2_units, layer1_units], "W2")
            b2 = self.__bias_bariable([layer2_units, 1], "b2")
            W3 = self.__weigfht_variable([self.n_y, layer2_units], "W3")
            b3 = self.__bias_bariable([self.n_y, 1], "b3")
        
        with tf.name_scope("layer1"):
            z1 = tf.add(tf.matmul(W1, self.X), b1)
            a1 = tf.nn.relu(z1)
        with tf.name_scope("layer2"):
            z2 = tf.add(tf.matmul(W2, a1), b2)
            a2 = tf.nn.relu(z2)
        with tf.name_scope("layer_output"):
            z3 = tf.add(tf.matmul(W3, a2), b3)
            a3 = tf.nn.softmax(z3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, name='A3')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.disc_norm_ep_reward)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        
    def __weigfht_variable(self, shape, name):
        initial = tf.contrib.layers.xavier_initializer(seed=1)
        return tf.get_variable(name, shape, initializer=initial)
    
    def __bias_bariable(self, shape, name):
        initial = tf.contrib.layers.xavier_initializer(seed=1)
        return tf.get_variable(name, shape, initializer=initial)
        
    def __disc_and_norm_rewards(self):   
        disc_norm_ep_rewards = np.zeros_like(self.episode_rewards)
        c = 0
        for t in reversed(range(len(self.episode_rewards))):
            c = c * self.reward_decay + self.episode_rewards[t]
            disc_norm_ep_rewards[t] = c

        disc_norm_ep_rewards -= np.mean(disc_norm_ep_rewards)
        disc_norm_ep_rewards /= np.std(disc_norm_ep_rewards)
        return disc_norm_ep_rewards


# In[4]:


class Monte_Carlo_Policy_Gradient():
    """
    Monte Carlo Policy Gradient method class
    """
    def __init__(self, env, num_episodes=200, learning_rate=0.01, reward_decay=0.95):
        
        self.nA = env.action_space.n
        self.nS = env.observation_space.shape[0]
        self.env = env
        self.num_episodes = num_episodes
        self.reward_decay = reward_decay
        self.learning_rate = learning_rate
        self.rewards = []
        self.RENDER_REWARD_MIN = 50
        self.RENDER_ENV = False
        self.PG = PolicyGradient(n_x=self.nS, n_y=self.nA, 
                                 learning_rate=self.learning_rate,
                                 reward_decay=self.reward_decay)
        
        # keep track of useful statistic
        record_head = namedtuple("Stats", ["episode_lengths","episode_rewards"])
        self.record = record_head(
                                episode_lengths = np.zeros(num_episodes),
                                episode_rewards = np.zeros(num_episodes))
        
    def mcpg_learn(self):
        """
        Monte Carlo Policy Gradient method core code. 
        """
        for i_episode in range(self.num_episodes):
            # print the number iter episode
            num_present = (i_episode+1) / self.num_episodes
            print("Episode {}/{}".format(i_episode + 1, self.num_episodes)) # end=""
            print("=" * round(num_present*60))
        
            # Reset the environment and pick the first action
            state = env.reset()
            reward = 0
            
            # One step in the environemt, replace code(while(True))
            for t in itertools.count():
                if self.RENDER_ENV: env.render()
                
                # step1: choose an action basoed on state
                action = self.PG.choose_action(state)
                
                # step2: take action in the environment
                next_state, reward, done, _ = env.step(action)
                
                # step3: store transition for training
                self.PG.store_transition(state, action, reward)
                
                # update statistics
                self.record.episode_rewards[i_episode] += reward
                self.record.episode_lengths[i_episode] = t
                
                if done:
                    episode_rewards_sum = sum(self.PG.episode_rewards)
                    self.rewards.append(episode_rewards_sum)
                    max_reward = np.amax(self.rewards)
                    
                    # step4: end of episode tran the PG network
                    _  = self.PG.learn()
                                    
                    print("reward:{}, max reward:{}, episode len:{}\n".format(episode_rewards_sum, max_reward, t))
                    if max_reward > self.RENDER_REWARD_MIN: self.RENDER_ENV = True
                    break
                    
                # step5: save new state
                state = next_state
        
        return self.record


# In[5]:


tf.reset_default_graph() 
mcpg = Monte_Carlo_Policy_Gradient(env, num_episodes=100)
result = mcpg.mcpg_learn()


# In[ ]:


import pandas as pd

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(13,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    # else:
    #     plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(13,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    # else:
    #     plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(13,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    # else:
    #     plt.show(fig3)
    plt.show()
    return fig1, fig2, fig3

plot_episode_stats(result)


# In[ ]:




