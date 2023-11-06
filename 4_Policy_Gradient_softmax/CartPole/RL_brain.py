import numpy as np 
import pandas as pd 
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

# reproducible
np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
   def __init__(self,
         
   )

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # fc1
        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        
        # use softmax to convert to probability
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            

    def choose_action(self):
      
    def store_transition(self):
      
    def learn(self):
    
    def _discount_and_norm_rewards(self):
      