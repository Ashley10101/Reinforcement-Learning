import gym
from RL_brain import Actor, Critic 
import tensorflow.compat.v1 as tf 
tf.disable_v2_behavior()

RENDER = False
MAX_EP_STEPS = 1000
DISPLAY_REWARD_THRESHOLD = 200
OUTPUT_GRAPH = False

env = gym.make('CartPole-v1', render_mode = "human")
env = env.unwrapped

sess = tf.Session()
actor = Actor(sess, n_features=env.observation_space.shape[0], n_actions=env.action_space.n, lr=0.001)
critic = Critic(sess, n_features=env.observation_space.shape[0], lr=0.01, gamma = 0.9)     # we need a good teacher, so the teacher should learn faster than the actor
sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)


for episode in range(3000):
    observation = env.reset()
    observation = observation[0]
    track_r = []
    t = 0
    while True:
        if RENDER: env.render()

        action = actor.choose_action(observation)
        observation_, reward, done, truncate, _ = env.step(action)

        if done or truncate:
            reward = -20
        track_r.append(reward)

        td_error = critic.learn(observation, reward, observation_) # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(observation, action, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        observation = observation_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", episode, "  reward:", int(running_reward))
            break

