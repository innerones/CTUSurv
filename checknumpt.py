import numpy as np
from glob import glob
import os
import torch
path = '/home/chipnpy_CHCAMS/'
feature = '/home/896_CHACAMS/896_CHACAMS_feature.npy'
image_list = glob(os.path.join(path, '*.npy'))
final_order = []
for image in image_list:
    crop_name = image.split('/')[-1].split('.')[0]
    final_order.append(crop_name)
path = '/home/final_nuclei/'
x_data = np.load(path + 'all_nuclei.npy', allow_pickle=True)
y_data = np.load(path + 'label.npy', allow_pickle=True)
feature = np.load(feature, allow_pickle=True)
feature = feature.tolist()
feature_list = torch.rand((1, 28, 28, 512)).cuda()
for index in final_order:
    temp = feature[index].unsqueeze(0)
    feature_list = torch.cat((feature_list, temp))
feature_list = feature_list[1:, :, :, :]
feature_list = feature_list.cpu()
np.save(path+'image_features.npy', feature_list)
np.save(path+'image_list.npy', final_order)
print(y_data)
