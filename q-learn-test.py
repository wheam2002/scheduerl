import numpy as np
import pandas as pd

worker_list = [['W1', 8, 16],['W2', 8, 16],['W3', 8, 16],['W4', 8, 16],['W5', 8, 16]]
worker = pd.DataFrame(data=worker_list, columns=['Worker', 'CPU', 'Mem'])

def genQueue():
    queue = pd.DataFrame(columns=['Task', 'Order', 'CPU', 'Mem', 'Image', 'Time', 'Start', 'End', 'Finished'])
    for i in range(2):
        pipeline = 'a' + str(i+1)
        a = [pipeline, 1, 2, 2, pipeline+'-1', 5, '', '', False]
        b = [pipeline, 2, 4, 4, pipeline+'-2', 10, '', '', False]
        c = [pipeline, 3, 4, 4, pipeline+'-3', 10, '', '', False]

        now = pd.DataFrame(data = [a, b, c], columns = ['Task', 'Order', 'CPU', 'Mem', 'Image', 'Time', 'Start', 'End', 'Finished'])
        queue = pd.concat([queue,now],ignore_index=True)
    return queue

def getQueue(queue):
    unfinished = queue['Finished'] == False
    queue_list = queue[unfinished]
    return queue_list

def genQtable(actions):
    table = pd.DataFrame(columns=actions)
    return table

def genActions(queue, worker_list):
    

    return 0

def envReward(action, knapsack):
    """返回下一步的state, reward和done
    """
    limit_W = 11
    knapsack_ = knapsack + [action] # 得到新的背包里的东西, 现在是[2,3], 向里面增加物品[1], 得到新的状态[1,2,3]
    knapsack_.sort()
    knapsack_W = np.sum([item['Weight'][i] for i in knapsack_]) # 计算当前包内物品的总和
    if knapsack_W > limit_W:
        r = -10
        done = True
    else:
        r = item['Value'][action]
        done = False
    return r, knapsack_, done

# print(genQueue())
# a = genQtable(['A','B'])
# a.loc[a.shape[0]]
# print(a)
# a.loc[a.shape[0]+1] = ['a3','b3']
# print(a)

print(genQueue())