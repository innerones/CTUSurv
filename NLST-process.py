import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as trans
NLST_path = '/home/nclei_seg/nlst_result/'
path = '/home/nclei_seg/processed_data/'
NLST_x = np.load(NLST_path + 'nuclei.npy', allow_pickle=True)
NLST_image = np.load(NLST_path + 'roi_feature.npy', allow_pickle=True)
NLST_label = np.load(NLST_path + 'lable.npy', allow_pickle=True)
NLST_x = NLST_x.tolist()
NLST_image = NLST_image.tolist()
NLST_label = NLST_label.tolist()
image = torch.rand(1, 28, 28, 512)
nuclei = torch.rand(1, 224, 224, 24)
i = 0
label = torch.rand(1, 2)
for patient_id in NLST_label:
    time_temp = NLST_label[patient_id]['time']
    status_temp = NLST_label[patient_id]['status']
    label_temp = torch.tensor((float(time_temp), float(status_temp))).unsqueeze(0)
    label = torch.cat((label, label_temp))
    print(i)
    i = i+1
    for image_id in NLST_image:
        p_id = image_id.split('_')[0]
        if p_id == patient_id:
            temp = NLST_image[image_id].unsqueeze(0).cpu()
            image = torch.cat((image, temp), 0)
    for nuclei_id in NLST_x:
        if nuclei_id == patient_id:
            temp = torch.from_numpy(NLST_x[nuclei_id]).unsqueeze(0).cpu()
            nuclei = torch.cat((nuclei, temp), 0)
nuclei = nuclei[1:, :, :, :].cuda()
image = image[1:, :, :, :]
label = label[1:, :]
nuclei_mask = torch.zeros_like(nuclei)
replace_x = nuclei.data
print(replace_x.shape)
for a in tqdm(range(449)):
    for b in range(224):
        for c in range(224):
            if replace_x[a, b, c, 0] == 0:
                    nuclei_mask[a, b, c, 0] = float(-100.0)
nuclei_mask = nuclei_mask[:, :, :, 0]
line_std = torch.rand((1)).cuda()
line_mean = torch.rand((1)).cuda()
all_nuclei = nuclei
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
np.save(path+'all_std.npy', line_std.cpu())
np.save(path+'all_mean.npy', line_mean.cpu())
np.savez(path+'nucleimask.npz', nuclei_mask=nuclei_mask.cpu())
mean = line_mean
mean = mean[1:]
std = line_std
std = std[1:]
x = nuclei
feature_list = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 22, 23]
# std = torch.from_numpy(std)
# mean = torch.from_numpy(mean)
normalizer = trans.Normalize(mean, std)
example = torch.rand((1, 224, 224, 24))
for i in tqdm(range(449)):
    temp_X = normalizer(x[i, :, :, :].squeeze(0).permute(2, 0, 1)).permute(1, 2, 0)
    example = torch.cat((example, temp_X.unsqueeze(0).cpu()))
example = example[1:, :, :, :]
np.save(path+'normalized_nuclei.npy', example.cpu())
np.save(path+'label.npy', label.cpu())
np.save(path+'image_features.npy', image.cpu())
print('ok')



