"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
"""

from env import Maze
from RL_brain import QLearningTable

EPISODE = 100
def update():
    for episode in range(EPISODE):
        observation = env.reset()
        while True:
            # 可视化环境
            env.render()
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                break
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    # .after(100, update)，在Tkinter中用于在指定时间（以毫秒为单位）后执行指定函数 update 的方式。
    RL = QLearningTable(actions=list(range(env.n_actions)))
    env.after(100, update)
    # 启动Tkinter主循环
    env.mainloop()