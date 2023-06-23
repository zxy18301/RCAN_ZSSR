# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:00:15 2023

@author: 93969
"""

import h5py


h5_fn = r'D:/Research_data/RCAN_HXN/RCAN_official/implementation/DIV_fly_scan_gray_eval_Mn_Zn_2.h5'
f = h5py.File(h5_fn, 'r')
# idx = 2
# idx = 10

mean_hr = []
mean_lr = []

for idx in range(len(f['hr'])):
    hr = f['hr'][str(idx)]
    hr = np.array(hr)
    lr = f['lr'][str(idx)]
    lr = np.array(lr)
    mean_hr.append(np.mean(hr))
    mean_lr.append(np.mean(lr))
    
print(np.mean(mean_hr))
print(np.mean(mean_lr))


mask = (hr-np.mean(mean_hr)) > 0
plt.figure()
plt.imshow(mask)