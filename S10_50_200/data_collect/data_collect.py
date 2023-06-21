# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 22:44:03 2023

@author: 93969
"""

import matplotlib.pyplot as plt
import os
os.chdir(r'D:\Research_data\RCAN_HXN\Natural_images_norm_scale2_8_finetune\xrf_S9\xrf')
from glob import glob
from skimage import io
import numpy as np
import random

# In[1]
'''
1. Collect data into tiff
'''
# 30 nm
scan_id = range(202969, 203239+1, 3)

# 120 nm
scan_id = range(203518, 203788+1, 3)

for i, scan in enumerate(scan_id):
    
    try:
        Mn_ = io.imread(rf'D:/Downloads/drive-download-20230221T071124Z-001/output_tiff_scan2D_{scan}/detsum_Mn_K_norm.tiff')
        Zn_ = io.imread(rf'D:/Downloads/drive-download-20230221T071124Z-001/output_tiff_scan2D_{scan}/detsum_Zn_K_norm.tiff')
    except:
        print(scan)
    
    if i == 0:
        Mn = np.zeros((len(scan_id), Mn_.shape[0], Mn_.shape[1]))
        Zn = np.zeros((len(scan_id), Zn_.shape[0], Zn_.shape[1]))
    
    Mn[i] = Mn_.copy()
    Zn[i] = Zn_.copy()
    
io.imsave(rf'D:\Research_data\RCAN_HXN\Natural_images_norm_scale2_8_finetune_S10\xrf_S10\Mn_30nm.tiff', np.float32(Mn))
io.imsave(rf'D:\Research_data\RCAN_HXN\Natural_images_norm_scale2_8_finetune_S10\xrf_S10\Zn_30nm.tiff', np.float32(Zn))

# In[2]
'''
2. Put all the data into folder
'''

def save_tiff(img_stack_path, train_path, eval_path, start_idx=0):
    img_stack = io.imread(img_stack_path)
    # path2save = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune/img_train_hr'
    
    for i in range(0, img_stack.shape[0]):
        img = img_stack[i]
        
        if (i >= 41 and i <= 50):
            io.imsave(rf'{eval_path}/{str(i+start_idx).zfill(3)}.tiff', np.float32(img))
        else:
            io.imsave(rf'{train_path}/{str(i+start_idx).zfill(3)}.tiff', np.float32(img))

Mn_60_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/xrf_S10/Mn_30nm.tiff'
Zn_60_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/xrf_S10/Zn_30nm.tiff'
Mn_240_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/xrf_S10/Mn_120nm.tiff'
Zn_240_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/xrf_S10/Zn_120nm.tiff'

hr_train_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_train_hr'
lr_train_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_train_lr'
hr_eval_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_eval_hr'
lr_eval_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_eval_lr'

save_tiff(Mn_60_path, hr_train_path, hr_eval_path, 0)
save_tiff(Zn_60_path, hr_train_path, hr_eval_path, 91)
save_tiff(Mn_240_path, lr_train_path, lr_eval_path, 0)
save_tiff(Zn_240_path, lr_train_path, lr_eval_path, 91)

# In[3]
'''
2 & 3: Save single tiff file into target folder with scan id
'''
scanid_hr = list(range(202969, 203239+1, 3))
scanid_lr = list(range(203518, 203788+1, 3))


scanid = list(zip(scanid_hr, scanid_lr))
random.shuffle(scanid)

scanid = np.array(scanid)
hr_train_scanid = scanid[:-10, 0]
lr_train_scanid = scanid[:-10, 1]

hr_eval_scanid = scanid[-10:, 0]
lr_eval_scanid = scanid[-10:, 1]

# save images
def savetiff2(path2save, scanid_list):

    for i, scanid in enumerate(scanid_list):
        # print(scanid)
        Zn = io.imread(rf'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/xrf_S10/xrf_S10/output_tiff_scan2D_{scanid}/detsum_Zn_K_norm.tiff')
        Mn = io.imread(rf'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/xrf_S10/xrf_S10/output_tiff_scan2D_{scanid}/detsum_Mn_K_norm.tiff')
        io.imsave(rf'{path2save}/{str(i).zfill(3)}_{scanid}_Zn.tiff', np.float32(Zn))
        io.imsave(rf'{path2save}/{str(i).zfill(3)}_{scanid}_Mn.tiff', np.float32(Mn))


hr_train_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_train_hr'
lr_train_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_train_lr'

hr_eval_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_eval_hr'
lr_eval_path = r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_eval_lr'

for path in [hr_train_path, lr_train_path, hr_eval_path, lr_eval_path]:
    if not os.path.exists(path):
        os.makedirs(path)
        
savetiff2(hr_train_path, hr_train_scanid)
savetiff2(lr_train_path, lr_train_scanid)
savetiff2(hr_eval_path, hr_eval_scanid)
savetiff2(lr_eval_path, lr_eval_scanid)


