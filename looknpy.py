import numpy as np
import torch
CHCAMS_path = '/home/final_nuclei/'
NLST_path = '/home/nclei_seg/processed_data/'
nuclei_CH = np.load(CHCAMS_path + 'normalized_nuclei.npy', allow_pickle=True)
nuclei_NL = np.load(NLST_path + 'normalized_nuclei.npy', allow_pickle=True)
label_CH = np.load(CHCAMS_path + 'label.npy').astype(int)
label_NL = np.load(NLST_path + 'label.npy').astype(int)
label_1 = torch.from_numpy(label_NL[:, 1]).unsqueeze(1)
label_2 = torch.from_numpy(label_NL[:, 0]).unsqueeze(1)
label_NL = torch.cat((label_1, label_2), 1).numpy()
image_CH = np.load(CHCAMS_path + 'image_features.npy', allow_pickle=True)
image_NL = np.load(NLST_path + 'image_features.npy', allow_pickle=True)
image_CH = torch.from_numpy(image_CH)
image_NL = torch.from_numpy(image_NL)
nuclei_mask_data = np.load('/home/single_nuclei/nucleimask.npz')
nuclei_mask_CH = nuclei_mask_data['nuclei_mask']
nuclei_mask_CH = torch.from_numpy(nuclei_mask_CH)
nuclei_mask_NL = np.load(NLST_path+'nucleimask.npz')
nuclei_mask_NL = nuclei_mask_NL['nuclei_mask']
nuclei_mask_NL = torch.from_numpy(nuclei_mask_NL).unsqueeze(3)
np.save(NLST_path+'nucleimask.npz', nuclei_mask_NL)
np.save(NLST_path+'label.npy', label_NL)
print('1')
