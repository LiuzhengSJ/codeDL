#!/usr/bin/env python
# coding: utf-8

# In[1]:

# import atari_py
import gym
import os
import sys
import itertools
import numpy as np
import random
import tensorflow as tf
from collections import defaultdict, namedtuple

import matplotlib
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

# print(atari_py.list_games())
# In[ ]:
# flags = tf.app.falgs        #   liuzheng

flags = tf.app.flags
FLAGS = flags.FLAGS


# Deep q Network
flags.DEFINE_boolean('use_gpu', True, 'Whether to use gpu or not. gpu use NHWC and gpu use NCHW for data_format')
flags.DEFINE_string('agent_type', 'DQN', 'The type of agent [DQN]')
flags.DEFINE_boolean('double_q', False, 'Whether to use double Q-learning')
flags.DEFINE_string('network_header_type', 'nips', 'The type of network header [mlp, nature, nips]')
flags.DEFINE_string('network_output_type', 'normal', 'The type of network output [normal, dueling]')

# Environment
flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('n_action_repeat', 1, 'The number of actions to repeat')
flags.DEFINE_integer('max_random_start', 30, 'The maximum number of NOOP actions at the beginning of an episode')
flags.DEFINE_integer('history_length0', 4, 'The length of history of observation to use as an input to DQN')
flags.DEFINE_integer('max_r', +1, 'The maximum value of clipped reward')
flags.DEFINE_integer('min_r', -1, 'The minimum value of clipped reward')
flags.DEFINE_string('observation_dims', '[80, 80]', 'The dimension of gym observation')
flags.DEFINE_boolean('random_start', True, 'Whether to start with random state')
flags.DEFINE_boolean('use_cumulated_reward', False, 'Whether to use cumulated reward or not')

# Training
flags.DEFINE_boolean('is_train0', True, 'Whether to do training or testing')
flags.DEFINE_integer('max_delta', None, 'The maximum value of delta')
flags.DEFINE_integer('min_delta', None, 'The minimum value of delta')
flags.DEFINE_float('ep_start', 1., 'The value of epsilon at start in e-greedy')
flags.DEFINE_float('ep_end', 0.01, 'The value of epsilnon at the end in e-greedy')
flags.DEFINE_integer('batch_size0', 32, 'The size of batch for minibatch training')
flags.DEFINE_integer('max_grad_norm', None, 'The maximum norm of gradient while updating')
flags.DEFINE_float('discount_r', 0.99, 'The discount factor for reward')

# Timer
flags.DEFINE_integer('t_train_freq', 4, '')

# Below numbers will be multiplied by scale
flags.DEFINE_integer('scale0', 10000, 'The scale for big numbers')
flags.DEFINE_integer('memory_size', 100, 'The size of experience memory (*= scale)')
flags.DEFINE_integer('t_target_q_update_freq', 1, 'The frequency of target network to be updated (*= scale)')
flags.DEFINE_integer('t_test', 1, 'The maximum number of t while training (*= scale)')
flags.DEFINE_integer('t_ep_end', 100, 'The time when epsilon reach ep_end (*= scale)')
flags.DEFINE_integer('t_train_max', 5000, 'The maximum number of t while training (*= scale)')
flags.DEFINE_float('t_learn_start', 5, 'The time when to begin training (*= scale)')
flags.DEFINE_float('learning_rate_decay_step', 5, 'The learning rate of training (*= scale)')

# Optimizer
flags.DEFINE_float('learning_rate0', 0.00025, 'The learning rate of training')
flags.DEFINE_float('learning_rate_minimum0', 0.00025, 'The minimum learning rate of training')
flags.DEFINE_float('learning_rate_decay0', 0.96, 'The decay of learning rate of training')
flags.DEFINE_float('decay', 0.99, 'Decay of RMSProp optimizer')
flags.DEFINE_float('momentum', 0.0, 'Momentum of RMSProp optimizer')
flags.DEFINE_float('gamma', 0.99, 'Discount factor of return')
flags.DEFINE_float('beta', 0.01, 'Beta of RMSProp optimizer')


# In[ ]:


# FLAGS.__delattr__()

# flags = tf.app.flags
# FLAGS = flags.FLAGS

flags.DEFINE_boolean("duele", False, "use dueling deep Q-learning")
flags.DEFINE_boolean("double", False, "use double Q-learning")

flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_integer("random_seed", 123, "value of random seed")
flags.DEFINE_boolean("display", False, "display the game")
flags.DEFINE_integer("scale", 10000, "step and the memory size")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("discount", 0.99,'discount ratio')
flags.DEFINE_float("learning_rate", 0.00025,'learning rate')
flags.DEFINE_float("learning_rate_min", 0.00025,'minimal learning rate')
flags.DEFINE_float("learning_rate_decay", 0.96,'maximal learning rate')
flags.DEFINE_integer("history_length", 4,'history length')
flags.DEFINE_integer("train_frequency", 4,'train freq')

flags.DEFINE_integer("learn_start", 50000,'learn start location')
flags.DEFINE_integer("frame_width", 84,'frame width')
flags.DEFINE_integer("frame_height", 84,'frame height')
flags.DEFINE_integer("max_reward", 1,'max reward')
flags.DEFINE_integer("min_reward", -1,'min reward')
flags.DEFINE_integer("episode_in_test", 80,'episode in train')
flags.DEFINE_integer("episode_in_train", 18000,'episode train')
flags.DEFINE_integer("test_max_step", 10000,'test max step')

FLAGS = flags.FLAGS


# In[ ]:


env = gym.make('Breakout-v0')
env = env.unwrapped
env.seed(FLAGS.random_seed)
tf.compat.v1.set_random_seed(FLAGS.random_seed)


# In[ ]:


class Environment(object):
    def __init__(self, env, history):
        self.env = env
        self.reward = 0
        self.terminal = False
        self.state_history = history
        self.state_dim = (FLAGS.frame_width, FLAGS.frame_height)
        self.nA = self.env.action_space.n
        self.nS = None

    def reset(self):
        self.env.reset()
    
    def random_start(self):
        self.reset()
        for _ in reversed(range(random.randint(4, 30))):
            state, _, _, _ = self.env.step(0)
            if 4 - _ > 0:
                self.state_history.push(self.__frame(state))
        
        self.env.render()
        return self.state_history
    
    def step(self, action):
        state, self.reward, self.terminal, _ = self.env.step(action)
        self.state = self.__frame(state)

        self.env.render()
        return self.state, self.reward, self.terminal
    
    @property
    def __frame(self, state):
        processed_state = np.array(state)
        frame_state = np.uint8(resize(rgb2gray(processed_state)/255., self.state_dim))
        return frame_state


# In[ ]:


class History(object):
    def __init__(self):
        self.history = np.zeros([FLAGS.history_length, 
                                FLAGS.frame_width,
                                FLAGS.frame_height], dtype=np.float32)
        
    def push(self, state):
        self.history[:-1] = self.history[1:]
        self.history[-1] = state
        
    def get(self):
        return self.history
        
    def clean(self):
        self.history *= 0


# In[ ]:


class Memory(object):
    def __init__(self):
        pass
    
    def push(self):
        pass
    
    def getState(self):
        pass
    
    def sample(self):
        pass


# In[ ]:


class Agent(object):
    def __init__(self, env, history, memory):
        self.env = env
        self.nA = env.nA
        self.state_history = history
        self.state_memroy = memory
        self.t = 0
        
        self.q_value, self.q_network = self.__build_network()
        self.target_q_value, self.target_q_network = self.__build_network()
        
        self.sess = tf.Session()
        
        tf.compat.v1.summary.FileWriter("summary/", self.sess.graph)
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
    
    def predict(self, state):
        if self.t < FLAGS.learn_start:
            action = random.randrange(self.nA)
        else:
            action = np.argmax(self.q_value.eval(feed_dict={self.s:state}))[0]
            
        return action
    
    def run(self, state, reward, action, done):
        reward = max(self.min_reward, min(self.self.max_reward, reward))
        
        self.history.add(state)
        self.memory.add(state, reward, action, done)
        
        if self.t > FLAGS.learn_start:
            if self.t % FLAGS.train_frequency:
                self.q_
        
        
        # 调用sess.run运行图，生成一步的训练过程数据  
        train_summary = sess.run(merge_summary, feed={})
        # 调用train_writer的add_summary方法将训练过程以及训练步数保存  
        train_writer.add_summary(train_summary, step)
        
        self.t += 1
    
    def __build_network(self):
        '''
        123
        build the natural network
        # Create placeholders
        '''
        with tf.compat.v1.name_scope('actor_inputs'):
            self.X = tf.compat.v1.placeholder(tf.float32,shape=[None, FLAGS.frame_width, FLAGS.frame_height, FLAGS.history_length],name="states")
            self.Y = tf.compat.v1.placeholder(tf.float32, shape=(self.n_y, None), name="action")
            self.disc_norm_ep_reward = tf.compat.v1.placeholder(tf.float32, name="td_error")

        with tf.compat.v1.name_scope("conv1"):
            conv1 = tf.compat.v1.nn.con2d(self.X, 32,
                                kernel_size=[8, 8], strides=[4, 4],
                                padding="same", activation=tf.nn.relu)
            # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
        with tf.compat.v1.name_scope("conv2"):
            conv2 = tf.compat.v1.nn.conv2(conv1, 64,
                               kernel_size=[4, 4], strides=[2, 2],
                               padding="same", activation=tf.compat.v1.nn.relu)
            # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
        with tf.compat.v1.name_scope("conv3"):
            conv3 = tf.compat.v1.nn.conv2(conv2, 64,
                               kernel_size=[3, 3], strides=[1, 1],
                               padding="same", activation=tf.compat.v1.nn.relu)
            # pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

        with tf.compat.v1.name_scope("dense_layer"):
            conv3_flat = tf.compat.v1.reshape(conv3, [-1,  * 64])
            dense1 = tf.compat.v1.layers.dense(conv3_flat, units=512, activation=tf.compat.v1.nn.relu)
        
        with tf.compat.v1.name_scope("logits_layer"):
            logits = tf.compat.v1.layers.dense(dense1, units=self.env.nA)
            

        with tf.compat.v1.name_scope('actor_loss'):
            neg_log_prob = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            loss = tf.compat.v1.reduce_mean(neg_log_prob * self.disc_norm_ep_reward)  # reward guided loss

        with tf.compat.v1.name_scope('actor_train'):
            self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)
    
    def __weigfht_variable(self, shape, name):
        initial = tf.compat.v1.contrib.layers.xavier_initializer(seed=1)
        return tf.compat.v1.get_variable(name, shape, initializer=initial)
    
    def __bias_bariable(self, shape, name):
        initial = tf.compat.v1.contrib.layers.xavier_initializer(seed=1)
        return tf.compat.v1.get_variable(name, shape, initializer=initial)
    
    def setup_summary(self):
        # 生成准确率标量图 
        average_reward = tf.placeholder('float32', None, name="average_reward")
        tf.summary.scalar("average_reward", average_reward)
        average_loss = tf.placeholder('float32', None, name="average_loss")
        tf.summary.scalar("average_loss", average_loss)
        average_q = tf.placeholder('float',None, name="average_q")
        tf.summary.scalar("average_q", average_q)
        
        episode_max_reward = tf.placeholder('float',None, name="episode_max_reward")
        tf.summary.scalar("episode_max_reward", episode_max_reward)
        episode_min_reward = tf.placeholder('float',None, name="episode_min_reward")
        tf.summary.scalar("episode_min_reward", episode_min_reward)
        episode_avg_reward = tf.placeholder('float',None, name="episode_avg_reward")
        tf.summary.scalar("episode_avg_reward", episode_avg_reward)
        episode_num = tf.placeholder('float',None, name="episode_num")
        tf.summary.scalar("episode_num", episode_num)
        episode_learning_rate = tf.placeholder('float',None, name="episode_learning_rate")
        tf.summary.scalar("episode_learning_rate", episode_learning_rate)
        
        # 定义一个写入summary的目标文件，dir为写入文件地址 
        merge_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./logs/', self.sess.graph)


# In[27]:


def deep_Qlearning(env):
    state_history = History()
    state_memory = Memory()
    env = Environment(gym.make('Breakout-v0'), state_history)
    agent = Agent(env, state_history, state_memory)

    if FLAGS.is_train:
        # for trainning the deep Q learning model
        max_avg_reward = 0
        
        for _ in range(FLAGS.episode_in_trains):
            total_reward, total_loss, total_q, ep_reward = 0, 0, 0, 0
            rewards, actions = [], []
            state_history = env.random_start()
            
            for t in itertools.count():
                # predict
                action = agent.predict(state_history.get())
                # action
                state, reward, done = env.step(action)
                # record
                # target = reward + gamma * np.amax(model.predict(next_state))
                agent.run(state, reward, action, done)
                
                if done:
                    ep_reward = 0
                    rewards.append(ep_reward)
                    state_history = env.random_start()
                else:
                    ep_reward += reward
                
                actions.append(action)
                total_reward += reward            
    else:
        # for test the deep Q learning model
        best_reward, best_idx = 0, 0
        for _ in range(FLAGS.episode_in_test):
            state_history = env.random_start()
            current_reward = 0
            
            for t in itertools.count():
                # predict
                action = agent.predict(state_history.get())
                # action
                state, reward, done = env.step(action)
                # record
                state_history.push(state)
                
                current_reward += reward
                if done: break
            
            # print out the reward 
            if current_reward > best_reward:
                best_reward = current_reward
                best_idx = _
                print("*"*80)
                print("[{}] Best reward:{}".format(best_idx, best_reward))
            


# In[12]:


random.randrange(5)


# In[ ]:




