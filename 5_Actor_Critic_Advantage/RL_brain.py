import numpy as np
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr = 0.001):
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self._buid_network()
    
    def _buid_network(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs = l1,
                units = self.n_actions,
                activation = tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

            with tf.variable_scope('exp_v'):
                log_prob = tf.log(self.acts_prob[0, self.a])
                self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)


    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})
        action = np.random.choice(np.arange(probs.shape[1]), p = probs.ravel())
        return action

class Critic(object):
    def __init__(self, sess, n_features, lr=0.01, gamma = 0.9):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.GAMMA = gamma 
        self.lr = lr
        self._build_network()
    
    def _build_network(self):
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

            with tf.variable_scope('squared_TD_error'):
                # TD(0)_error = (r+gamma*V_next) - V_eval
                self.td_error = self.r + self.GAMMA  * self.v_ - self.v
                self.loss = tf.square(self.td_error)

            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                        {self.s: s, self.v_: v_, self.r: r})
        return td_error


