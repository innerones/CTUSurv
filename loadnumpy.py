import numpy
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as trans
path = '/home/final_nuclei/'
mean = np.load(path+'all_mean.npy')
mean = mean[1:]
std = np.load(path+'all_std.npy')
std = std[1:]
x_data = np.load(path+'all_nuclei.npy', allow_pickle=True)
x = torch.from_numpy(x_data)
feature_list = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23]
# x = x[:, :, :, (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23)]
std = torch.from_numpy(std)
mean = torch.from_numpy(mean)
normalizer = trans.Normalize(mean, std)
example = torch.rand((1,224,224,24))
for i in tqdm(range(686)):
    temp_X = normalizer(x[i, :, :, :].squeeze(0).permute(2, 0, 1)).permute(1, 2, 0)
    example = torch.cat((example, temp_X.unsqueeze(0)))
example = example[1:, :, :, :]
np.save(path+'normalized_nuclei.npy', example.cpu())
print('ok')
# std = std[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23]
# mean = mean[0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23]
# all_temp = torch.rand((686, 224, 224))
# for i in tqdm(feature_list):
#     temp_line = x[:, :, :, int(i)]
#     temp_line = temp_line.where(x != 0, (x-mean[int(i)])/std[int(i)], x)
#     all_temp = torch.cat((all_temp, temp_line))
# print('ok')
