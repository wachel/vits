import os
import sys

name_to_subtitle = {}
with open('data/cp2077/subtitle.txt') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line != '':
            ws = line.split('\t')
            if len(ws) == 2:
                name, subtitle = ws
                name_to_subtitle[name] = subtitle

train_list = []
val_list = []
paths = ['data/cp2077/wav/johnny','data/cp2077/wav/v_q_f','data/cp2077/wav/v_q_m']
for i, path in enumerate(paths):
    for j, fn in enumerate(os.listdir(path)):
        fp = os.path.join(path, fn)
        key = os.path.splitext(fn)[0]
        if key in name_to_subtitle:
            subtitle = name_to_subtitle[key]
            if j < 40:
                val_list.append(f'{fp}|{i}|{subtitle}')
            else:
                train_list.append(f'{fp}|{i}|{subtitle}')

with open('data/cp2077/cp2077_train_filelist.txt', 'wt') as f:
    f.write('\n'.join(train_list))

with open('data/cp2077/cp2077_val_filelist.txt', 'wt') as f:
    f.write('\n'.join(val_list))    
