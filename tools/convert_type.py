import os
import torch
import numpy as np


def convert_pytorch_to_numpy(root_path:str, end_with):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(end_with):
                pt_path = os.path.join(root, file)
                npy_path = pt_path.replace('.pt', '.npy')
                x = torch.load(pt_path).cpu().numpy()
                np.save(npy_path, x)

convert_pytorch_to_numpy('/home/bigfoot/gits/data', 'w2v_hm.pt')