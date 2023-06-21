# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 01:06:00 2022

@author: 93969
"""

import argparse
import os
os.chdir(r'D:\Research_data\RCAN_HXN\RCAN_official\implementation')
import copy
import numpy as np
from PIL import Image

from tensorboardX import SummaryWriter
import os
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
# from predict_from_png.py import tensor2numpy

def tensor2numpy(inputs):
    
    inputs = inputs.squeeze(0).cpu().numpy()
    inputs = inputs[0, ...]
    
    return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default=r'D:/Research_data/RCAN_HXN/RCAN_official/implementation/DIV_fly_scan_gray_bicubic_test_norm_mean.h5')
    parser.add_argument('--eval-file', type=str, default=r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_align/DIV_fly_scan_gray_eval_bicubic_test_norm_mean.h5')
    parser.add_argument('--outputs-dir', type=str, default=r'./outputs_scale_bicubic_l1_2_correctPSNR')
    parser.add_argument('--weights-file', type=str)

    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-workers', type=int, default=0)
    
    
    # parser.add_argument('--model', default='RCAN',
    #                 help='model name')
    # Model
    parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups') # 10
    parser.add_argument('--rgb_range', type=int, default=256,
                help='number of residual groups') 
    parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
    parser.add_argument('--scale', type=int, default='4',
                    help='super resolution scale')
    parser.add_argument('--n_resblocks', type=int, default=20,
                        help='number of residual blocks') # 20
    parser.add_argument('--n_colors', type=int, default=1,
                            help='number of channel')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps') # 64
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
    
    # if args.weights_file is not None:
    #     state_dict = model.state_dict()
    #     for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
    #         print(n, p)
    #         if n in state_dict.keys():
    #             state_dict[n].copy_(p)
    #         else:
    #             raise KeyError(n)

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dataset = TrainDataset(args.train_file, patch_size=args.patch_size, scale=args.scale)
    # dataset_visualization(train_dataset, 250)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    
    log_loss = []
    log_psnr = []

    for epoch in range(args.num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * (0.5 ** (epoch // int(args.num_epochs * 0.25)))

        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}  lr: {}'.format(epoch, args.num_epochs - 1, param_group['lr']))

            for data in train_dataloader:
                inputs, labels = data


                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                # break
                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))
                

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))
            log_loss.append(epoch_losses.avg)

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)

            # preds = convert_rgb_to_y(denormalize(preds.squeeze(0)), dim_order='chw')
            # labels = convert_rgb_to_y(denormalize(labels.squeeze(0)), dim_order='chw')

            # preds = preds[args.scale:-args.scale, args.scale:-args.scale]
            # labels = labels[args.scale:-args.scale, args.scale:-args.scale]
            preds = tensor2numpy(preds)
            
            inputs = tensor2numpy(inputs)
            labels = tensor2numpy(labels)
            epoch_psnr.update(PSNR(preds, labels), len(inputs))

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        log_psnr.append(epoch_psnr.avg)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))
    np.save('log_loss.npy', log_loss)
    np.save('log_psnr.npy', log_psnr)
    
    
    
    #
    # plt.figure()
    # plt.imshow(inputs)
    # plt.axis('off')
    
    # plt.figure()
    # plt.imshow(labels)
    # plt.axis('off')

    # plt.figure()
    # plt.imshow(preds)
    # plt.axis('off')
    