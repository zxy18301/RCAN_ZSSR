# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 01:06:00 2022

@author: 93969
"""

import argparse
import os
import copy
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import RCAN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, convert_rgb_to_y, denormalize, dataset_visualization, tensor2numpy, PSNR
import matplotlib.pyplot as plt
import cv2
from skimage import io
from fly_scan_aligned_2 import fly_scan_aligned_crop2
# from predict_from_png.py import tensor2numpy

def tensor2numpy(inputs):
    
    inputs = inputs.squeeze(0).cpu().numpy()
    inputs = inputs[0, ...]
    
    return inputs

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default=r'./DIV_fly_scan_gray_bicubic_test.h5')
    parser.add_argument('--eval-file', type=str, default=r'./DIV_fly_scan_gray_eval_bicubic_test.h5')
    parser.add_argument('--outputs-dir', type=str, default=r'./outputs_scale_bicubic_test_l1')
    # parser.add_argument('--weights-file', type=str, default=r'D:/Research_data/RCAN_HXN/RCAN_official/Natural_2_8_pystack_align_submean_finetune/outputs_scale_2_8_l1_finetune_submean_freeze2/x4/x4/best.pth')
    parser.add_argument('--weights-file', type=str, default=r'D:/Research_data/RCAN_HXN/RCAN_official/implementation/outputs_scale_2_8_mse_finetune_submean/x4/x4/best.pth')
    parser.add_argument('--image_gt', type=str, default=r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune/img_eval_hr/056.tiff')
    parser.add_argument('--image_lr', type=str, default=r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune/img_eval_lr/056.tiff')
    
    # parser.add_argument('--image_gt', type=str, default=r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_eval_hr/009_203239.tiff')
    # parser.add_argument('--image_lr', type=str, default=r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/img_eval_lr/009_203788.tiff')

    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-workers', type=int, default=0)
    
    
    # parser.add_argument('--model', default='RCAN',
    #                 help='model name')
    # Model
    parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
    parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
    parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
    parser.add_argument('--scale', type=int, default='4',
                    help='super resolution scale')
    parser.add_argument('--n_resblocks', type=int, default=20,
                        help='number of residual blocks')
    parser.add_argument('--n_colors', type=int, default=1,
                            help='number of channel')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')

    # parser.add_argument('--use_fast_loader', action='store_true')
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RCAN(args).to(device)
    
    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            # print(n, p)
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)


    hr = io.imread(args.image_gt, as_gray=True)
    lr = io.imread(args.image_lr, as_gray=True)
    
    hr = NormalizeData(hr) - 0.0706
    lr = NormalizeData(lr) - 0.0706
    lr = np.expand_dims(lr, 2)
    hr = np.expand_dims(hr, 2)
    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0)
    hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0)
    lr = torch.from_numpy(lr).to(device)
    hr = torch.from_numpy(hr).to(device)

    model.eval()

    with torch.no_grad():
        preds = model(lr)


    preds = tensor2numpy(preds)
    
    hr = tensor2numpy(hr)
    lr = tensor2numpy(lr)

    
    
    print(PSNR(hr, preds, 1))
    plt.figure()
    plt.imshow(lr)
    plt.axis('off')
    
    plt.figure()
    plt.imshow(preds)
    plt.axis('off')

    plt.figure()
    plt.imshow(hr)
    plt.axis('off')
    
    
    plt.show()
    
    # hr = io.imread(args.image_gt)
    
    # hr_up4 = cv2.resize(hr, (hr.shape[1]*4, hr.shape[0]*4))
    # hr_up4 = NormalizeData(hr_up4)
    # plt.figure()
    # plt.imshow(hr_up4)
    
    # prb_int = np.load(r'D:/Research_data/RCAN_HXN/RCAN_official/Natural_2_8_pystack_align_submean_finetune/recon_202978_t1_mode_probe_ave_int_16x16.npy')
    # hr_up4_fly_scan = fly_scan_aligned_crop2(hr_up4, prb_int, 4, 3)
    # preds_fly_scan = fly_scan_aligned_crop2(preds, prb_int, 4, 3)
    
    # plt.figure()
    # plt.imshow(hr_up4_flyu_scan)
    
    # plt.figure()
    # plt.imshow(preds_fly_scan)
    # # plt.figure()
    # # plt.imshow(hr_up4 - preds)
    
    # plt.figure()
    # plt.imshow(hr_up4[480-60:480+60, 640-60:640+60])
    
    # plt.figure()
    # plt.imshow(preds[480-60:480+60, 640-60:640+60])
    
  
