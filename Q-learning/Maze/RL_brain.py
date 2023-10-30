import pandas as pd
import numpy as np

class QLearningTable:
    def __init__(self, actions, epsilon = 0.1, gamma = 0.9, lr=0.01):
        '''
            epsilon: the parameter of epsilon-greedy
            gamma: the decay rate
            lr: learning rate
        '''
        self.actions = actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.Q_table = pd.DataFrame(columns=self.actions, dtype = np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() <= self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.Q_table.loc[observation,:]
            # random的目的是在相同值当中随便选一个
            action = np.random.choice(state_action[state_action==np.max(state_action)].index)
        return action
    
    def learn(self, observation, action, reward, observation_):
        self.check_state_exist(observation_)
        q_predict = self.Q_table.loc[observation, action]
        if observation_ != 'terminal':
            q_target = reward + self.gamma*self.Q_table.loc[observation_, :].max()
        else:
            q_target = reward
        self.Q_table.loc[observation, action] += self.lr * (q_target - q_predict)
        
    
    def check_state_exist(self, state):
        if state not in self.Q_table.index:
            self.Q_table = self.Q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.Q_table.columns,
                    name = state
                )
            )
    