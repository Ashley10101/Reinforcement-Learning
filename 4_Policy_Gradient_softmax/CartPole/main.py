import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400
RENDER = False  # rendering wastes time  

env = gym.make('CartPole-v1', render_mode = "human")
# env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    # output_graph=True,
)

for i_episode in range(3000):
    observation = env.reset()
    observation = observation[0]
    while True:
        if RENDER: env.render()
        action = RL.choose_action(observation)
        observation_, reward, done, truncate, info = env.step(action)
        RL.store_transition(observation, action, reward)

        if done or truncate:
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            # if i_episode == 0:
            #     plt.plot(vt)    # plot the episode vt
            #     plt.xlabel('episode steps')
            #     plt.ylabel('normalized state-action value')
            #     plt.show()
            # break
        observation = observation_


