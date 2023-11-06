import gym
from RL_brain import DeepQNetwork

env = gym.make('CartPole-v1', render_mode = "human") 
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.001,)

total_step = 0

# import importlib
# package_location = importlib.util.find_spec(gym).submodule_search_locations[0]
# print(package_location)

for episode in range(100):
    observation = env.reset()
    observation = observation[0]
    ep_r = 0
    while True:
        env.render()
        action = RL.choose_action(observation)
        # print(env.step(action))
        # print('======')
        observation_, reward, done, truncated, _ = env.step(action)
        # the smaller theta and closer to center the better
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)
        if total_step > 1000:
            RL.learn()
        ep_r += reward
        if done or truncated:
            print(
                'episode: ', episode,
                'ep_r: ', round(ep_r, 2),
                'epsilon: ', round(RL.epsilon, 2)
            )
            break
        
        observation  = observation_
        total_step += 1

RL.plot_cost()