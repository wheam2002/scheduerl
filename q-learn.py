import pandas as pd
import numpy as np
import copy


item = pd.DataFrame(data=[['A1', 2, 4, True], ['A2', 4, 4, False], ['A3', 6, 8, False], ['A4', 8, 10, False], ['A5', 10, 12, False]], columns=['Pod', 'CPU', 'Memory', 'Pending'])
item2 = pd.DataFrame(data=[['B1', 2, 4, True], ['B2', 4, 4, False], ['B3', 6, 8, False], ['B4', 8, 10, False], ['B5', 10, 12, False]], columns=['Pod', 'CPU', 'Memory', 'Pending'])

all_task = pd.concat([item,item2]).reset_index(drop=True)

node_list = ['N1','N2','N3','N4']
node_usage = ['N1','N2','N3','N4']
limit_CPU = 8
limit_MEM = 16
gamma = 0.9
knapsack = []
# a = []

cpu, Memory, time, all_task

[2,2,5,'A'],[3,3,10,'A']

a = np.array([[2, 4, 8], [4, 4, 10], [6, 8, 30], [6, 8, 30], [6, 8, 15], [8, 8, 170]])
# b = copy.deepcopy(a)s

a = [np.append(a[i], ['N1']) for i in range(len(a))]
print(a)
# for i in node_list:
#     for j in a:
#         j.append(i)
# print(a)
# for i in range(10):
#     for j in a:
#         j.append()
#     a = a+[[2, 4, 8], [4, 4, 10], [6, 8, 30], [6, 8, 30], [6, 8, 15], [8, 8, 170]]
# action = list(range(len(a)))



def getQueue():
    queue_list = pd.DataFrame()
    queue_list = all_task[all_task['Pending'] == True]
    return queue_list
    # queue_list = getQueue()

def getAction():
    action_list = []
    for i in queue_list.itertuples():
        for j in node_list:
            action_list.append([j,i.Pod])
    return action_list

def getEnv():

    

    return env

def envReward(action, knapsack, queue_list):
    
    knapsack.append(action)
    knapsack.sort()
    
    for i in knapsack:
        
        pod_filt = (queue_list['Pod'] == i[1])
        pod = queue_list.loc[pod_filt]
        
        pod_cpu = int(pod['CPU'].values)
        pod_mem = int(pod['Memory'].values)

    return knapsack

# print(envReward(['N4', 'B1'], [['N2', 'A1']], queue_list))

def mu_policy(Q, epsilon, nA, observation, actions):
    
    actionsList = list(set(actions).difference(set(observation))) # 可以挑选的动作
    # 看到这个state之后, 采取不同action获得的累计奖励
    action_values = Q.loc[str(observation), :]
    # 使用获得奖励最大的那个动作
    greedy_action = action_values.idxmax()
    # 是的每个动作都有出现的可能性
    probabilities = np.zeros(nA)
    for i in actionsList:
        probabilities[i] = 1/len(actionsList) * epsilon
    probabilities[greedy_action] = probabilities[greedy_action] + (1 - epsilon)
    return probabilities

def pi_policy(Q, observation):
    '''
    这是greedy policy, 每次选择最优的动作.
    其中: 
    - Q是q table, 为dataframe的格式;
    '''
    action_values = Q.loc[str(observation), :]
    best_action = action_values.idxmax() # 选择最优的动作
    return np.eye(len(action_values))[best_action] # 返回的是两个动作出现的概率

def check_state(Q, knapsack, actions):
    '''检查状态knapsack是否在q-table中, 没有则进行添加
    '''
    if str(knapsack) not in Q.index: # knapsack表示状态, 例如现在包里有[1,2]
        # append new state to q table
        q_table_new = pd.Series([np.NAN]*len(actions), index=Q.columns, name=str(knapsack))
        # 下面是将能使用的状态设置为0, 不能使用的设置为NaN (这个很重要)
        for i in list(set(actions).difference(set(knapsack))):
            q_table_new[i] = 0
        return Q.append(q_table_new)
    else:
        return Q

def qLearning(actions, num_episodes, discount_factor=1.0, alpha=0.7, epsilon=0.2):
    # 环境中所有动作的数量
    nA = len(actions)
    # 初始化Q表
    Q = pd.DataFrame(columns=actions)
    # 记录reward和总长度的变化
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes+1),
        episode_rewards=np.zeros(num_episodes+1))
    for i_episode in range(1, num_episodes + 1):
        # 开始一轮游戏
        knapsack = [] # 开始的时候背包是空的
        Q = check_state(Q, knapsack, actions)
        action = np.random.choice(nA, p=mu_policy(Q, epsilon, nA, knapsack, actions)) # 从实际执行的policy, 选择action
        for t in itertools.count():
            reward, next_knapsack, done = envReward(action, knapsack) # 执行action, 返回reward和下一步的状态
            Q = check_state(Q, next_knapsack, actions)
            next_action = np.random.choice(nA, p=mu_policy(Q, epsilon, nA, next_knapsack, actions)) # 选择下一步的动作
            # 更新Q
            Q.loc[str(knapsack), action] = Q.loc[str(knapsack), action] + alpha*(reward + discount_factor*Q.loc[str(next_knapsack), :].max() - Q.loc[str(knapsack), action])
            # 计算统计数据(带有探索的策略)
            stats.episode_rewards[i_episode] += reward # 计算累计奖励
            stats.episode_lengths[i_episode] = t # 查看每一轮的时间
            if done:
                break
            if t > 10:
                break
            knapsack = next_knapsack
            action = next_action
        if i_episode % 50 == 0:
            # 打印
            print('\rEpisode {}/{}. | '.format(i_episode, num_episodes), end='')
    return Q, stats