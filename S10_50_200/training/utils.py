# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 01:12:57 2022

@author: 93969
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from math import log10, sqrt


def convert_rgb_to_y(img, dim_order='hwc'):
    if dim_order == 'hwc':
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    else:
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.


def denormalize(img):
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img


def preprocess(img, device):
    img = np.array(img).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(img)
    x = ycbcr[..., 0]
    x /= 255.
    x = torch.from_numpy(x).to(device)
    x = x.unsqueeze(0).unsqueeze(0)
    return x, ycbcr


def calc_psnr(img1, img2, max=255.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def step_scan(img, prb, step):
    # prb is a  complex array while prb_int is the abs**2 of prb
    

    img_r = np.flipud(img[:,:,0] / np.max(img[:,:,0])).T
    img_g = np.flipud(img[:,:,1] / np.max(img[:,:,1])).T
    img_b = np.flipud(img[:,:,2] / np.max(img[:,:,2])).T
    nx, ny = np.shape(img_r)
    
    
    nx_prb,ny_prb = np.shape(prb)
    prb_int = np.abs(prb)**2
    
    # step = 4
    xrf_nx = int(np.floor((nx - nx_prb) / step))
    xrf_ny = int(np.floor((ny - ny_prb) / step))
    
    
    xrf_r = np.zeros((xrf_nx,xrf_ny))
    xrf_g = np.zeros((xrf_nx,xrf_ny))
    xrf_b = np.zeros((xrf_nx,xrf_ny))
    
    for ii in range(xrf_nx):
        for jj in range(xrf_ny):
            iy_s = int(step * jj)
            ix_s = int(step * ii)
            xrf_r[ii,jj] = np.sum(img_r[ix_s:ix_s+nx_prb,iy_s:iy_s+ny_prb] * prb_int)
            xrf_g[ii,jj] = np.sum(img_g[ix_s:ix_s+nx_prb,iy_s:iy_s+ny_prb] * prb_int)
            xrf_b[ii,jj] = np.sum(img_b[ix_s:ix_s+nx_prb,iy_s:iy_s+ny_prb] * prb_int)
    
    xrf = (np.dstack((np.fliplr(xrf_r).T/np.max(xrf_r),np.fliplr(xrf_g).T/np.max(xrf_g),np.fliplr(xrf_b).T/np.max(xrf_b))) * 256) .astype(np.uint8)

    
    return xrf

def step_scan_gray(img, prb, step):

    img_r = np.flipud(img[:,:] / np.max(img[:,:])).T
    nx, ny = np.shape(img_r)
    
    
    nx_prb,ny_prb = np.shape(prb)
    prb_int = np.abs(prb)**2
    
    # step = 4
    xrf_nx = int(np.floor((nx - nx_prb) / step))
    xrf_ny = int(np.floor((ny - ny_prb) / step))
    
    
    xrf_r = np.zeros((xrf_nx,xrf_ny))
    
    for ii in range(xrf_nx):
        for jj in range(xrf_ny):
            iy_s = int(step * jj)
            ix_s = int(step * ii)
            xrf_r[ii,jj] = np.sum(img_r[ix_s:ix_s+nx_prb,iy_s:iy_s+ny_prb] * prb_int)
    
    
    xrf = np.uint8((np.fliplr(xrf_r).T/np.max(xrf_r)) * 256)
    
    return xrf

def dataset_visualization(dataset, idx):
    lr, hr = dataset[str(idx)]
    plt.figure()
    lr = lr.transpose(1,2,0)
    hr = hr.transpose(1,2,0)
    plt.figure()
    plt.subplot(121)
    plt.imshow(hr)
    plt.title('hr')
    plt.subplot(122)
    plt.imshow(lr)
    plt.title('lr')
    
def PSNR(original, compressed, max_pixel):
    
    original = np.array(original)
    compressed = np.array(compressed)
    
    # original = np.float64(original*255)
    # compressed = np.float64(compressed*255)
    
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    # max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def tensor2numpy(inputs):
    
    inputs = inputs.squeeze(0).cpu().numpy()
    inputs = inputs[0, ...]
    
    return inputs