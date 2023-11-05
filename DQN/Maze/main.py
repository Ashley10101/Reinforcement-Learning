from env import Maze
from RL_brain import DeepQNetwor

def run_maze():
    step = 0
    for episode in range(300):
        observation = env.reset()
        while True:
            env.render()
            action = RL.choose_action(observation)

            observation_, reward, done = env.step(action)

            # experience replay
            RL.store_transition(observation, action, reward, observation_)

            # Since RL need trajectories to learn, so we set it start 
            # learning after 200 epochs and every 5 epochs
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            
            observation = observation_

            if done:
                break
            step += 1

    print('game over')
    env.destroy()

if __name__ == '__main__':
    env = Maze()
    RL = DeepQNetwor(env.n_actions, )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()