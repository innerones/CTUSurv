import os
import torch
from tqdm import tqdm
path = '/home/duo_swin/selected_attention/'
for index in range(548):
    list_x = []
    list_y = []
    mix_x = []
    mix_y = []
    for i in range(6):
        temp_x = torch.load(path+str(index)+'_single_x_'+str(i)+'.pt')
        list_x.append(temp_x)
    for i in range(3):
        temp_y = torch.load(path+str(index)+'_single_y_'+str(i)+'.pt')
        list_y.append(temp_y)
    for i in range(3):
        temp_mix_x = torch.load(path + str(index) + '_mix_x_'+str(i)+'.pt')
        mix_x.append(temp_mix_x)
    for i in range(3):
        temp_mix_y = torch.load(path + str(index) + '_mix_y_'+str(i)+'.pt')
        mix_y.append(temp_mix_y)
    for i in list_x:
        B_, numheads, numpatch, _ = i.shape
        i = torch.mean(i, dim=1)
        # i = torch.mean(i, dim=1)
        i = torch.max(i, dim=1)
        if index == 10:
            print('1')
        print('shape')


