# """
# Tensorflow: 2.14
# Python: 3.7
# """
#!/usr/bin/env python3
import tensorflow as tf
print('hello')

# class DeepQNetwork:
#     def __init__(
#             self, 
#             n_actions,
#             n_features,
#             learning_rate = 0.01,
#             reward_decay = 0.9,
#             e_greedy = 0.9,
#             replace_target_iter = 300,
#             memory_size = 500,
#             batch_size = 32,
#             e_greedy_increment = None,
#             output_graph = False,
#             ):
#         self.n_actions = n_actions
#         self.n_features = n_features
#         self.lr = learning_rate
#         self.gamma = reward_decay
#         self.epsilon_max = e_greedy
#         self.replace_target_iter = replace_target_iter
#         self.memory_size = memory_size
#         self.batch_size = batch_size
#         self.epsilon_increment = e_greedy_increment
#         self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

#     def _build_net(self):
#         # ------------------- build evaluate_net -----------------
#         self.s = tf.placeholder(tf.float32, [None, self.n_features], name = 's') # input
#         self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name = 'Q-target') # for evaluate the loss
#         with tf.variable_scope('eval_net'):
#             # c_names(collections_names) are the collections to store variables
#             c_names, n_l1, w_initializer, b_initializer = \
#             ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10,  \
#             tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1) 

#             # first layer. collections is used later when assign to target net
#             with tf.variable_scope('l1'):
#                 w1 = tf.variable('w1', [self.n_features, n_l1], initializer = w_initializer, collections=c_names)
#                 b1 = tf.variable('b1', [1, n_l1], initializer = b_initializer, collections=c_names)
#                 l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
            
#             # second layer. collections is used later when assign to target net
#             with tf.variable_scope('l2'):
#                 w2 = tf.variable('w2', [n_l1, self.n_actions], initializer = w_initializer, collections=c_names)
#                 b2 = tf.variable('b2', [1, self.n_actions], initializer = b_initializer, collections=c_names)
#                 l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
        
#             with tf.variable_scope('loss'):
#                 self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
#             with tf.variable_scope('train'):
#                 self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
#         # ------------------- build target_net -----------------

#     def store_transition(self, s, a, r, s_):

#     def choos_action(self, observation):
    
#     def learn(self):

#     def plot_cost(self):
        
