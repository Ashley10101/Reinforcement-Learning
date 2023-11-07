import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0',render_mode = "human")
env = env.unwarpped 
RENDER = False
DISPLAY_REWARD_THRESHOLD = -2000
RL = PolicyGradien(
    n_actions = env.action_space.n,
    n_features = env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
)

for episode in range(1000):
    observation = env.reset()
    observation = observation[0]
    while True:
        if RENDER: env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, truncate, _ = env.step(action)
        RL.store_transition(observation, action, reward)
        if done or truncate:
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", episode, "  reward:", int(running_reward))
            # if episode == 30:
            #     plt.plot(vt)  # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            vt = RL.learn()
            break
        observation = observation_