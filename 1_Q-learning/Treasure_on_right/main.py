from RL_brain import *
from env import *

np.random.seed(2)
N_STATES = 6
ACTIONS = ['left', 'right']
MAX_EPISODES = 13
GAMMA = 0.9 # discount factor
ALPHA = 0.1 # learning rate

def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminal = False
        update_env(N_STATES, state, episode, step_counter)
        while not is_terminal:
            action = choose_action(state, q_table)
            state_, reward = get_env_feedback(N_STATES, state, action)
            q_predict = q_table.loc[state, action]
            if state_ != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[state_,:].max()
            else:
                q_target = reward
                is_terminal = True
            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            state = state_
            step_counter += 1
            update_env(N_STATES, state, episode, step_counter)
    return q_table

if __name__ == "__main__":
    q_table = rl()