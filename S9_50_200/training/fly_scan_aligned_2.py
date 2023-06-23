#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 19:46:16 2022

@author: xiaoyin
"""

import numpy as np
import matplotlib.pyplot as plt
# import simulate_zp as szp
from skimage import io
import sys
import os
import pathlib
from PIL import Image
# from fly_scan import fly_scan
import cv2
from pystackreg import StackReg




# fly scan

def fly_scan(img, prb_int, step):
    
    
    img = np.array(img)
    # img_pad = np.pad(img, ((2,2), (2,2)))
    padding = prb_int.shape[0] // 2
    
    img_pad = np.pad(img, (padding, padding), 'constant')
    

    img_r = np.flipud(img_pad[:,:] / np.max(img[:,:])).T
    nx, ny = np.shape(img.T)
    nx_prb, ny_prb = np.shape(prb_int)
    

    xrf_nx = int(np.floor((nx - nx_prb) / step))
    xrf_ny = int(np.floor((ny - ny_prb) / step))
    
    # xrf_nx = int(np.floor(nx / step))
    # xrf_ny = int(np.floor(ny / step))
    
    
    xrf_fly_r = np.zeros((xrf_nx,xrf_ny))
    
    
    for ii in range(xrf_nx):
        for jj in range(xrf_ny):
            iy_s = int(step * jj)
            ix_s = int(step * ii)
            for kk in range(step):
                if kk == 0:
                    tmp_r = img_r[ix_s+kk: ix_s+kk+nx_prb,iy_s: iy_s+ny_prb] * prb_int
    
                else:
                    tmp_r += img_r[ix_s+kk: ix_s+kk+nx_prb,iy_s: iy_s+ny_prb] * prb_int
    
            xrf_fly_r[ii,jj] = np.sum(tmp_r) / step
            # print(ii, jj, xrf_fly_r[ii,jj])
    
    
    xrf_fly = ((np.fliplr(xrf_fly_r).T/np.max(xrf_fly_r)) * 256)
    # xrf_fly = ((xrf_fly_r/np.max(xrf_fly_r)) * 256) .astype(np.uint8)
    return xrf_fly / np.max(xrf_fly)

def fly_scan_padding(img, prb_int, step):
    
    
    # Test
    # step = 4
    # img = Image.open(r'D:/Research_data/Image/275.png').convert('L')
    # prb_int = np.load(r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_4/recon_202978_t1_mode_probe_ave_int_16x16.npy')
    
    img = np.array(img)
    
    # padding = prb_int.shape[0] // 2
    w, h = img.shape
    img_pad = np.pad(img, ((w//2, w//2), (h//2, h//2)), 'constant')
    
    img_r = np.flipud(img_pad[:,:] / np.max(img_pad[:,:])).T
    
    nx, ny = np.shape(img_r)
    nx_prb, ny_prb = np.shape(prb_int)

    xrf_nx = int(np.floor((nx - nx_prb) / step))
    xrf_ny = int(np.floor((ny - ny_prb) / step))
    
    # xrf_nx = int(np.floor(nx / step))
    # xrf_ny = int(np.floor(ny / step))

    xrf_fly_r = np.zeros((xrf_nx,xrf_ny))
    
    for ii in range(xrf_nx):
        for jj in range(xrf_ny):
            iy_s = int(step * jj)
            ix_s = int(step * ii)
            for kk in range(step):
                if kk == 0:
                    tmp_r = img_r[ix_s+kk: ix_s+kk+nx_prb,iy_s: iy_s+ny_prb] * prb_int
    
                else:
                    tmp_r += img_r[ix_s+kk: ix_s+kk+nx_prb,iy_s: iy_s+ny_prb] * prb_int
    
            xrf_fly_r[ii,jj] = np.sum(tmp_r) / step
            # print(ii, jj, xrf_fly_r[ii,jj])
    
    
    xrf_fly = ((np.fliplr(xrf_fly_r).T/np.max(xrf_fly_r)) * 256)
    xrf_fly /= np.max(xrf_fly)
    # plt.figure()
    # plt.imshow(xrf_fly)
    return xrf_fly

def find_border_idx(row, length2sum):
    # row = col.copy()
    # length2sum = hr.shape[1]//2
    s = 0
    for i in range(len(row) // 2):
        temp = np.sum(row[i:i+length2sum])
        if temp > s:
            pos = i
            s = temp
    
    return pos

def fly_scan_aligned(hr, prb_int, step):
    xrf2 = fly_scan_padding(hr, prb_int, step)
    row = np.sum(xrf2, axis=0) #248, 763
    col = np.sum(xrf2, axis=1) # 165, 515
    
    
    pos_row = find_border_idx(row, hr.shape[1]//step)
    pos_col = find_border_idx(col, hr.shape[0]//step)
    
    xrf2_ = xrf2[pos_col: pos_col+hr.shape[0]//step, pos_row: pos_row+hr.shape[1]//step]
    
    return xrf2_

def resize2integer(hr, scale):
    hr = np.array(hr)
    w, h = hr.shape[0] // scale * scale, hr.shape[1] // scale * scale
    hr = cv2.resize(hr, (h-1, w-1), interpolation=cv2.INTER_CUBIC)
    
    return hr

def fly_scan_aligned_crop(hr, prb_int, step, cut):
    
    xrf4 = fly_scan_aligned(hr, prb_int, step)
    
    xrf4_0 = np.sum(xrf4, axis=0)
    xrf4_1 = np.sum(xrf4, axis=1)
    
    col1, col2 = 0, xrf4.shape[1]
    row1, row2 = 0, xrf4.shape[0]
    if xrf4_0[0] < xrf4_0[1] * cut:
        col1 = 1
    if xrf4_0[-1] < xrf4_0[-2] * cut:
        col2 = -1
    if xrf4_1[0] < xrf4_1[1] * cut:
        row1 = 1
    if xrf4_1[-1] < xrf4_1[-2] * cut:
        row2 = -1
    print(col1, col2, row1, row2)
    
    xrf4_crop = xrf4[row1: row2, col1: col2]

    return xrf4_crop

def fly_scan_aligned_crop2(hr, prb_int, step, cut):
    
    xrf4 = fly_scan_aligned(hr, prb_int, step)
    
    xrf4_0 = np.sum(xrf4, axis=0)
    xrf4_1 = np.sum(xrf4, axis=1)
    
    col1, col2 = 0, xrf4.shape[1]
    row1, row2 = 0, xrf4.shape[0]
    if np.abs(xrf4_0[1] - xrf4_0[0]) > (np.abs(xrf4_0[3] - xrf4_0[2]) + np.abs(xrf4_0[2] - xrf4_0[1])) / 2 * cut:
        col1 = 1
    if np.abs(xrf4_0[-2] - xrf4_0[-1]) > (np.abs(xrf4_0[-4] - xrf4_0[-3]) + np.abs(xrf4_0[-3] - xrf4_0[-2])) / 2 * cut:
        col2 = -1
    if np.abs(xrf4_1[1] - xrf4_1[0]) > (np.abs(xrf4_1[3] - xrf4_1[2]) + np.abs(xrf4_1[2] - xrf4_1[1])) / 2 * cut:
        row1 = 1
    if np.abs(xrf4_1[-2] - xrf4_1[-1]) > (np.abs(xrf4_1[-4] - xrf4_1[-3]) + np.abs(xrf4_1[-3] - xrf4_1[-2])) / 2 * cut:
        row2 = -1
    # print(col1, col2, row1, row2)
    
    xrf4_crop = xrf4[row1: row2, col1: col2]

    return xrf4_crop

'''
Begin
'''
# Get prb function -- prb_int
if __name__=='__main__':
    prb_int = np.load(r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_4/recon_202978_t1_mode_probe_ave_int_16x16.npy')
    
    
    hr = Image.open(r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8/Image_eval/3722.png').convert('L') # 2880 800
    hr = np.array(hr)
    # hr = resize2integer(hr, 8)
    
    prb_half_size = int(prb_int.shape[0]/2)
    
    
    
    ##def fly_scan_aligned_crop(hr, prb_int, step, cut)
    xrf4 = fly_scan_aligned(hr, prb_int, 2)
    # xrf4, step, cut
    step = 2
    cut = 3
    xrf4_0 = np.sum(xrf4, axis=0)
    xrf4_1 = np.sum(xrf4, axis=1)
    

    col1, col2 = 0, xrf4.shape[1]
    row1, row2 = 0, xrf4.shape[0]
    # if xrf4_0[0] < xrf4_0[1] * cut:
    #     col1 = 1
    # if xrf4_0[-1] < xrf4_0[-2] * cut:
    #     col2 = -1
    # if xrf4_1[0] < xrf4_1[1] * cut:
    #     row1 = 1
    # if xrf4_1[-1] < xrf4_1[-2] * cut:
    #     row2 = -1
    # print(col1, col2, row1, row2)
    
    if np.abs(xrf4_0[1] - xrf4_0[0]) > (np.abs(xrf4_0[3] - xrf4_0[2]) + np.abs(xrf4_0[2] - xrf4_0[1])) / 2 * cut:
        col1 = 1
    if np.abs(xrf4_0[-2] - xrf4_0[-1]) > (np.abs(xrf4_0[-4] - xrf4_0[-3]) + np.abs(xrf4_0[-3] - xrf4_0[-2])) / 2 * cut:
        col2 = -1
    if np.abs(xrf4_1[1] - xrf4_1[0]) > (np.abs(xrf4_1[3] - xrf4_1[2]) + np.abs(xrf4_1[2] - xrf4_1[1])) / 2 * cut:
        row1 = 1
    if np.abs(xrf4_1[-2] - xrf4_1[-1]) > (np.abs(xrf4_1[-4] - xrf4_1[-3]) + np.abs(xrf4_1[-3] - xrf4_1[-2])) / 2 * cut:
        row2 = -1
    print(col1, col2, row1, row2)
    
    
    xrf4_crop = xrf4[row1: row2, col1: col2]
    
    ## return xrf4_crop
    hr = cv2.resize(hr, (xrf4_crop.shape[1] * step, xrf4_crop.shape[0] * step), interpolation=cv2.INTER_CUBIC)
    # hr_crop = hr[prb_half_size//2: -prb_half_size//2, prb_half_size//2: -prb_half_size//2]
    # hr_crop = cv2.resize(hr_crop, (xrf4_crop.shape[1] * step, xrf4_crop.shape[0] * step), interpolation=cv2.INTER_CUBIC)
    # xrf2 = fly_scan_aligned(hr, prb_int, 2)
    # xrf8 = fly_scan_aligned(hr, prb_int, 8)
    
    # xrf2 = xrf8.copy()
    # # row
    # row = np.sum(xrf2, axis=0) #248, 763
    # col = np.sum(xrf2, axis=1) # 165, 515
    
    # pos_row = find_border_idx(row, hr.shape[1]//8)
    # pos_col = find_border_idx(col, hr.shape[0]//8)
    
    # xrf2_ = xrf2[pos_col: pos_col+hr.shape[0]//8, pos_row: pos_row+hr.shape[1]//8]
            
    plt.figure()
    plt.imshow(hr)
    plt.axis('off')
    
    

    
    
    plt.figure()
    plt.imshow(xrf4)
    plt.axis('off')
    

    
    plt.figure()
    plt.imshow(xrf4_crop)
    plt.axis('off')
    

    

    # xrf4_crop_up = cv2.resize(xrf4_crop, (hr_crop.shape[1], hr_crop.shape[0]))
    # plt.figure()
    # plt.imshow(xrf4_crop_up - hr_crop)
    # plt.axis('off')
    
    # plt.figure()
    # plt.imshow(hr_crop)
    # plt.axis('off')
    
    # sr = StackReg(StackReg.TRANSLATION)
    # out_tra = sr.register_transform(xrf4_crop_up, hr_crop) # ref, mov
    # plt.figure()
    # plt.imshow(out_tra)
    # plt.axis('off')
# ====================================================================================================
    hr = Image.open(r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8/Image_eval/3722.png').convert('L') # 2880 800
    hr = np.array(hr)
    xrf2 = fly_scan_aligned_crop2(hr, prb_int, 2, 3)
    xrf8 = fly_scan_aligned_crop2(hr, prb_int, 8, 3)
    
    # hr_crop = hr[prb_half_size//2: , prb_half_size//2: ]
    # hr_crop = hr[prb_half_size//4: , prb_half_size//4: ]
    # xrf8 = fly_scan_aligned_crop(hr, prb_int, 8, 0.7)
    
    # xrf_2_8 = fly_scan_aligned_crop(xrf2, prb_int, 4, 0.7)
    
    
    plt.figure()
    plt.imshow(hr)
    plt.axis('off')
    
    plt.figure()
    plt.imshow(xrf2)
    plt.axis('off')
    
    plt.figure()
    plt.imshow(xrf8)
    plt.axis('off')
# =====================================================================================================
    xrf4_aligned = fly_scan_aligned_crop2(hr, prb_int, 4, 3)
    xrf4 = fly_scan(hr, prb_int, 4)
    
    
    plt.figure()
    plt.imshow(xrf4)
    plt.axis('off')
    
    plt.figure()
    plt.imshow(xrf4_aligned)
    plt.axis('off')
    
    

    

    
    # plt.figure()
    # plt.imshow(xrf8)
    # plt.axis('off')
    
    # plt.figure()
    # plt.imshow(xrf_2_8)
    # plt.axis('off')
    