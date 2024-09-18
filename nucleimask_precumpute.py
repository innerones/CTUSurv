import numpy as np
from tqdm import tqdm
import torch
path = '/home/final_nuclei/'
x_data = np.load(path + 'all_nuclei.npy', allow_pickle=True)
x = torch.from_numpy(x_data)
B, H, W, C = x.shape
nuclei_mask = torch.zeros((B, H, W, 1))
replace_x = x.data
print(replace_x.shape)
for a in tqdm(range(B)):
    for b in range(H):
        for c in range(W):
            if replace_x[a, b, c, 0] == 0:
                    nuclei_mask[a, b, c, 0] = float(-100.0)
print(nuclei_mask)
np.savez('nucleimask.npz', nuclei_mask=nuclei_mask)