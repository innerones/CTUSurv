import os

import torch
import numpy as np
from tqdm import tqdm
path = '/home/final_nuclei/'
nuclei = 'all_nuclei.npy'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# temp_line = torch.rand((1)).cuda()
line_std = torch.rand((1)).cuda()
line_mean = torch.rand((1)).cuda()
all_nuclei = np.load(path+nuclei)
all_nuclei = torch.from_numpy(all_nuclei).cuda()
# for i in tqdm(range(24)):
#     for a in tqdm(range(686)):
#         for b in range(224):
#             for c in range(224):
#                 temp_num = all_nuclei[a, b, c, i]
#                 if temp_num == 0:
#                     continue
#                 else:
#                     temp_line = torch.cat((temp_line, temp_num.unsqueeze(0)))
#     temp_line = temp_line[1:]
#     line_std = torch.cat((line_std, torch.std(temp_line).unsqueeze(0)))
#     line_mean = torch.cat((line_mean, torch.mean(temp_line).unsqueeze(0)))
#     temp_line = torch.rand((1)).cuda()
# np.save(path+'all_std.npy', line_std)
# np.save(path+'all_mean.npy', line_mean)
for i in tqdm(range(24)):
    temp_nuclei_line = torch.tensor(all_nuclei[:, :, :, i]).cuda()
    # indice = torch.nonzero(temp_nuclei_line).cuda()
    # for index in tqdm(indice):
    #     a, b, c = index[0], index[1], index[2]
    #     temp_nuclei = temp_nuclei_line[a,b,c]
    #     temp_line = torch.cat((temp_line, temp_nuclei.unsqueeze(0)))
    mask = temp_nuclei_line.ne(0)
    temp_line = torch.masked_select(temp_nuclei_line, mask)
    line_std = torch.cat((line_std, torch.std(temp_line).unsqueeze(0)))
    line_mean = torch.cat((line_mean, torch.mean(temp_line).unsqueeze(0)))
    print('done')
np.save(path+'all_std.npy', line_std.cpu())
np.save(path+'all_mean.npy', line_mean.cpu())





