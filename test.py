import copy
import pandas as pd
# original_list = [['carrots', "apple"], 'kiwi', 'grapes', 'cherry']

# print("The original list is")
# print(original_list)

# copied_list = copy.copy(original_list)

# original_list[0] = ['banana', "apple"]
# original_list[1] = 'orange'

# print("The original list after modification is")
# print(original_list)
# print("The copied list is")
# print(copied_list)
queue = pd.DataFrame(columns=['Task','Order','CPU','Mem','Time'])
for i in range(2):
    pipeline = 'a' + str(i+1)
    a = [pipeline, 1, 2, 2, 5, '', '', False]
    b = [pipeline, 2, 4, 4, 10, '', '', False]
    c = [pipeline, 3, 4, 4, 10, '', '', False]

    now = pd.DataFrame(data = [a, b, c], columns = ['Task', 'Order', 'CPU', 'Mem', 'Time', 'Start', 'End', 'Finished'])
    queue = pd.concat([queue,now],ignore_index=True)

