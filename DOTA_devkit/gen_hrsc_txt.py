import os
import sys

path = 'data/HRSC2016/Test/Annotations'
path_pre = os.path.dirname(path)
name_list = [os.path.basename(i).split('.')[0] for i in os.listdir(path)]

with open(os.path.join(path_pre, 'text.txt'), 'w') as f:
    for i in name_list:
        f.write(i)
        f.write('\n')
