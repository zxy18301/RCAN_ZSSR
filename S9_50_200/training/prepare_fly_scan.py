# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 13:09:42 2022

@author: 93969
"""

import argparse
import glob
import h5py
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from skimage import io

import os
os.chdir(r'D:\Research_data\RCAN_ZSSR\S9_50_200\training')

import sys
# sys.path.append(os.getcwd())
# from utils import step_scan_gray

import glob
import matplotlib.pyplot as plt
import random
# from fly_scan import fly_scan
import cv2
from pystackreg import StackReg


'''
functions
'''
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Check the image pairs in h5 file
def random_crop(lr, hr, size, scale):
    lr_left = random.randint(0, lr.shape[1] - size - 1)
    # lr_left = 0
    lr_right = lr_left + size
    lr_top = random.randint(0, lr.shape[0] - size - 1)
    # lr_top = 0
    lr_bottom = lr_top + size
    hr_left = lr_left * scale
    hr_right = lr_right * scale
    hr_top = lr_top * scale
    hr_bottom = lr_bottom * scale
    lr = lr[lr_top:lr_bottom, lr_left:lr_right]
    hr = hr[hr_top:hr_bottom, hr_left:hr_right]
    return lr, hr

def align_hr_lr(hr, lr):
    lr_reg = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
    sr = StackReg(StackReg.AFFINE)
    hr = sr.register_transform(lr_reg, hr)    
    return hr, lr

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image-hr-path', type=str, default=r'D:\Research_data\RCAN_ZSSR\S9_50_200\data_collect\img_train_hr')
    parser.add_argument('--image-lr-path', type=str, default=r'D:\Research_data\RCAN_ZSSR\S9_50_200\data_collect\img_train_lr')
    parser.add_argument('--eval-hr-path', type=str, default=r'D:\Research_data\RCAN_ZSSR\S9_50_200\data_collect\img_eval_hr')
    parser.add_argument('--eval-lr-path', type=str, default=r'D:\Research_data\RCAN_ZSSR\S9_50_200\data_collect\img_eval_lr')
    parser.add_argument('--output_path', type=str, default=r'.\S9_crop_norm_training.h5')
    parser.add_argument('--output-path-eval', type=str, default=r'.\S9_crop_norm_eval.h5')
    parser.add_argument('--check-flag', type=int, default=0)
    # parser.add_argument('--step-size', type=int, default=4)
    # parser.add_argument('--probe-path', type=str, default='recon_202978_t1_mode_probe_ave_int_16x16.npy')
    parser.add_argument('--random-num', type=int, default=4)
    parser.add_argument('--num_cropping', type=int, default=4)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--patch-size', type=int, default=48)
    # parser.add_argument('--patch_size', type=int, default=64)
    args = parser.parse_args()
    
    
    

    
    '''
    Train
    '''
    
    hr_list = sorted(glob.glob(f'{args.image_hr_path}/*'))
    lr_list = sorted(glob.glob(f'{args.image_lr_path}/*'))
    
    
    h5_file = h5py.File(args.output_path, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    
    patch_idx = 0
    test = 0
    
        
    for i in range(len(hr_list)):

        hr = io.imread(hr_list[i])
        lr = io.imread(lr_list[i])
        
        hr, lr = align_hr_lr(hr, lr)
        # hr = np.pad(hr, ((40, 40), (0, 0)), mode='minimum')
        # lr = np.pad(lr, ((10, 10), (0, 0)), mode='minimum')

        
        if test:
            
            plt.figure()
            plt.imshow(hr)
            plt.axis('off')
            
            plt.figure()
            plt.imshow(lr)
            plt.axis('off')
            
            plt.figure()
            plt.imshow(hr_padded)
            plt.axis('off')
            
            plt.figure()
            plt.imshow(lr_padded)
            plt.axis('off')
            

        
        # data augmentation here
        for j in range(args.random_num):
            for i in range(args.num_cropping):
                if j == 0:
                    
                    hr_ = hr.copy()
                    lr_ = lr.copy()
                    hr_ = NormalizeData(hr_)
                    lr_ = NormalizeData(lr_)
                    
                    lr_, hr_ = random_crop(lr_, hr_, args.patch_size, args.scale)
                    
                    
                    hr_ = np.expand_dims(hr_, 2)
                    lr_ = np.expand_dims(lr_, 2)
                
    
                    hr_group.create_dataset(str(patch_idx), data=hr_)
                    lr_group.create_dataset(str(patch_idx), data=lr_)
                    
                    patch_idx+=1
                    
                elif j == 1:
                    
                    # horizontal flip
                    lr_ = lr[:, ::-1].copy()
                    hr_ = hr[:, ::-1].copy()
                    
                    lr_ = NormalizeData(lr_)
                    hr_ = NormalizeData(hr_)
                    
                    lr_, hr_ = random_crop(lr_, hr_, args.patch_size, args.scale)
                    
                    hr_ = np.expand_dims(hr_, 2)
                    lr_ = np.expand_dims(lr_, 2)
                
                    lr_group.create_dataset(str(patch_idx), data=lr_)
                    hr_group.create_dataset(str(patch_idx), data=hr_)
                    
                    patch_idx+=1
                    
                elif j == 2:
                    
                    # vertical flip
                    lr_ = lr[::-1, :].copy()
                    hr_ = hr[::-1, :].copy()
                    
                    lr_ = NormalizeData(lr_)
                    hr_ = NormalizeData(hr_)
                    
                    lr_, hr_ = random_crop(lr_, hr_, args.patch_size, args.scale)
                    
                    hr_ = np.expand_dims(hr_, 2)
                    lr_ = np.expand_dims(lr_, 2)
                
                    lr_group.create_dataset(str(patch_idx), data=lr_)
                    hr_group.create_dataset(str(patch_idx), data=hr_)
                    
                    patch_idx+=1
                    
                else:
                    
                    # rotate 90
                    lr_ = np.rot90(lr, axes=(1, 0)).copy()
                    hr_ = np.rot90(hr, axes=(1, 0)).copy()
                    
                    lr_ = NormalizeData(lr_)
                    hr_ = NormalizeData(hr_)
                    
                    lr_, hr_ = random_crop(lr_, hr_, args.patch_size, args.scale)
                    
    
                    hr_ = np.expand_dims(hr_, 2)
                    lr_ = np.expand_dims(lr_, 2)
                
                    lr_group.create_dataset(str(patch_idx), data=lr_)
                    hr_group.create_dataset(str(patch_idx), data=hr_)
                    
                    patch_idx+=1
        
        print(i, patch_idx, hr_.shape, lr_.shape)
        
    
    h5_file.close()
    
    '''
    Eval
    '''
    
    hr_list = sorted(glob.glob(f'{args.eval_hr_path}/*'))
    lr_list = sorted(glob.glob(f'{args.eval_lr_path}/*'))
    
    h5_file = h5py.File(args.output_path_eval, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    patch_idx = 0
    
    for i in range(len(hr_list)):

        hr = io.imread(hr_list[i])
        lr = io.imread(lr_list[i])
        
        hr = hr[40:200, 40:240]
        lr = lr[10:50, 10:60]
        
        hr, lr = align_hr_lr(hr, lr)
        # hr = np.pad(hr, ((40, 40), (0, 0)), mode='minimum')
        # lr = np.pad(lr, ((10, 10), (0, 0)), mode='minimum')

        hr_ = hr.copy()
        lr_ = lr.copy()
        
        hr_ = NormalizeData(hr_)
        hr_ = np.expand_dims(hr_, 2)
        lr_ = NormalizeData(lr_)
        lr_ = np.expand_dims(lr_, 2)
    

        hr_group.create_dataset(str(patch_idx), data=hr_)
        lr_group.create_dataset(str(patch_idx), data=lr_)
        
        patch_idx+=1
        
        print(i, patch_idx, hr_.shape, lr_.shape)
        
    
    h5_file.close()
    
    
    
    if args.check_flag:
        h5_fn = r'D:/Research_data/RCAN_ZSSR/S9_50_200/training/S9_crop_norm_eval.h5'
        f = h5py.File(h5_fn, 'r')
        idx = 4
        # idx = 10
        hr = np.array(f['hr'][str(idx)])
        lr = np.array(f['lr'][str(idx)])
        plt.figure()
        # plt.subplot(121)
        plt.imshow(hr)
        plt.title('hr')
        plt.axis('off')
        # plt.subplot(122)
        plt.figure()
        plt.imshow(lr)
        plt.title('lr')
        plt.axis('off')        
        # plt.figure()
        # plt.imshow((hr - 0.0703) > 0)
        
        # plt.figure()
        # plt.imshow(hr)
        
        f.close()
