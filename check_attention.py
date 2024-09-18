import os
import torch
path = '/home/duo_swin/attention/'
savepath = '/home/duo_swin/selected_attention/'
label_path = '/home/duo_swin/'
label = torch.load(label_path+'label.pt')
attention_list = os.listdir(path)
print(len(label))
print(len(attention_list))
attention_list.sort()
count = 0
for attention in attention_list:
    temp_count = count % 52
    label_count = count // 52
    count = count+1
    print(label_count)
    if temp_count == 4:
        name = str(label_count)+'_single_x_0.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 10:
        name = str(label_count) + '_single_x_1.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 14:
        name = str(label_count) + '_single_x_2.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 18:
        name = str(label_count) + '_single_x_3.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 22:
        name = str(label_count) + '_single_y_0.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 24:
        name = str(label_count) + '_mix_x_0.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 25:
        name = str(label_count) + '_mix_y_0.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 32:
        name = str(label_count) + '_single_x_4.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 38:
        name = str(label_count) + '_single_y_1.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 40:
        name = str(label_count) + '_mix_x_1.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 41:
        name = str(label_count) + '_mix_y_1.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 44:
        name = str(label_count) + '_single_x_5.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 46:
        name = str(label_count) + '_single_y_2.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 48:
        name = str(label_count) + '_mix_x_2.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)
    if temp_count == 49:
        name = str(label_count) + '_mix_y_2.pt'
        temp = torch.load(path+attention)
        torch.save(temp, savepath+name)



