import numpy as np
import pandas as pd

EPSILON = 0.1
ACTIONS = ['left', 'right']
np.random.seed(2)

def build_q_table(n_states, actions = ACTIONS):
    table = pd.DataFrame(np.zeros((n_states, len(actions))), columns = actions)
    return table

def choose_action(state, q_table):
    state_actions = q_table.iloc[state,:]
    # .all() 是 Pandas DataFrame 和 Series 中的一个方法，用于检查所有元素是否为 True。
    if (np.random.uniform()<EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax() #找到具有最大 Q 值的动作对应的索引位置
    return action_name
    

table = build_q_table(3, ['left', 'right'])
choose_action(0, table)
# print(table)