import time
FRESH_TIME = 0.1

def get_env_feedback(n_state, state, action):
    if action == "right":
        if state == n_state - 2:
            reward = 1
            state_ = 'terminal'
        else:
            state_ = state + 1
            reward = 0
    else:
        reward = 0
        state_ = max(state - 1, 0)
    return state_, reward

def update_env(n_state, state, episode, step_counter):

    '''
    \r 是回车字符，它将光标移到当前行的开头。
    {} 是占位符，用于将变量的值插入到字符串中。
    format(interaction) 用于将 interaction 的值插入到字符串中。
    end='' 参数指定在 print 函数结束后不要添加换行符，以便在同一行上继续输出。
    这段代码的效果是，在同一行上动态显示 interaction 的值，而不是创建多个行来显示不同的值。这对于实时更新的进度条或其他动态信息非常有用
    '''

    env_list = ['-']*(n_state-1) + ['T']
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

