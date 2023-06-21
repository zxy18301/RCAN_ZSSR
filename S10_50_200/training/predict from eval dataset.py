import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import RCAN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, convert_rgb_to_y, denormalize, dataset_visualization, PSNR
import matplotlib.pyplot as plt

def tensor2numpy(inputs):
    
    inputs = inputs.squeeze(0).cpu().numpy()
    inputs = inputs[0, ...]
    
    return inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train-file', type=str, default=r'D:\Research_data\RCAN\Natural_Images_probe_CentralCrop\DIV_fly_scan_gray.h5')
    parser.add_argument('--eval-file', type=str, default=r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune/DIV_fly_scan_gray_eval_2.h5')
    # parser.add_argument('--outputs-dir', type=str, default=r'D:\Research_data\RCAN\Natural_Images_probe_CentralCrop\outputs_scale_4')
    parser.add_argument('--weights-file', type=str, default=r'D:/Research_data/RCAN_HXN/RCAN_official/Natural_2_8_pystack_align_submean_finetune/outputs_scale_2_8_l1_finetune_submean_freeze2/x4/x4/best.pth')
    # parser.add_argument('--weights-file', type=str, default=r'D:/Research_data/RCAN_HXN/Natural_images_norm_scale2_8_finetune_S10/outputs_scale_2_finetune_l1/x4/x4/best.pth')
    parser.add_argument('--scale', type=int, default=4)  
    parser.add_argument('--num_features', type=int, default=64)
    parser.add_argument('--num_rg', type=int, default=10)
    parser.add_argument('--num_rcab', type=int, default=20)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()

    # args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    # args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    # if not os.path.exists(args.outputs_dir):
    #     os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = RCAN(args).to(device)
    
    if args.weights_file is not None:
        state_dict = model.state_dict()
        for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                raise KeyError(n)
            
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    
    i = 0
    for data in eval_dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds1 = model(inputs)
            # preds2 = model(labels)
            # preds3 = model(preds1)
            
        i += 1
        if i == 2:
            break
    

    
    inputs = tensor2numpy(inputs)
    preds1 = tensor2numpy(preds1)
    # preds2 = tensor2numpy(preds2)
    # preds3 = tensor2numpy(preds3)
    labels = tensor2numpy(labels)
    
    plt.figure()
    
    plt.subplot(131)
    plt.imshow(inputs)
    plt.title('Blurred image patch', fontsize=24, fontweight='bold')
    
    # plt.figure()
    plt.subplot(132)
    plt.imshow(preds1)
    plt.title('Prediction from blurred image patch', fontsize=24, fontweight='bold')
    
    
    # plt.figure()
    plt.subplot(133)
    plt.imshow(labels)
    plt.title('Ground Truth', fontsize=24, fontweight='bold')
    
    print(PSNR(labels, preds1))
    plt.show()
    
    print('123')