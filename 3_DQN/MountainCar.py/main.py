import gym
from RL_brain import DeepQNetwork

env = gym.make('MountainCar-v0', render_mode = "human") 
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=3,
                n_features=2,
                learning_rate=0.001, e_greedy=0.9,
                replace_target_iter=300, memory_size=3000,
                e_greedy_increment=0.0002,)

total_step = 0

# import importlib
# package_location = importlib.util.find_spec(gym).submodule_search_locations[0]
# print(package_location)

for episode in range(10):
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
        position, velocity = observation_

        # the higher the better
        reward = abs(position - (-0.5))     # r in [0, 1]

        RL.store_transition(observation, action, reward, observation_)
        if total_step > 1000:
            RL.learn()
        ep_r += reward
        if done or truncated:
            get = '| Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print('Epi: ', episode,
                get,
                '| Ep_r: ', round(ep_r, 4),
                '| Epsilon: ', round(RL.epsilon, 2))
            break
        
        observation  = observation_
        total_step += 1

RL.plot_cost()