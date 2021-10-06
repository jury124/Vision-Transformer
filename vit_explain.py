import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from vit_rollout import VITAttentionRollout

import os
import gc
import time
import wandb
import logging 
import json
import numpy as np
import pandas as pd

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms 
from torch.cuda.amp import GradScaler, autocast
from torchsummary import summary

from model.dataset import CustomDataset
from model.ViT import ViT

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def visual(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ViT( 
        in_channels = args.in_channels, 
        patch_size = args.patch_size, 
        emb_size = args.emb_size, 
        img_size = args.img_size, 
        depth = args.depth, 
        n_classes = args.n_classes
        ).to(device)

    checkpoint = torch.load(f'{args.model_path}/checkpoint.pth.tar')
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)
    del checkpoint

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    
    # dataset_test = CustomDataset(data_path=args.data_path,
    #                              transform=transform_test, 
    #                              phase='test')
    # dataloader_test = DataLoader(dataset_test, drop_last=False,
    #                              batch_size=1, shuffle=True, pin_memory=True
    #                              )

    file_list = os.listdir('./data/test/')
    for i in range(len(file_list)):
        img = Image.open(f'./data/test/{file_list[i]}').convert('RGB')
        img = img.resize((224, 224))
        input_tensor = transform(img).unsqueeze(0)
        img_tensor = input_tensor.to(device, non_blocking=True)

        print("Doing Attention Rollout")
        attention_rollout = VITAttentionRollout(model, head_fusion='max', discard_ratio=0.9)
        mask = attention_rollout(img_tensor)
        name = "attention_rollout_{:.3f}_{}.png".format(0.9,'max')


        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)

        d = pd.read_csv('./test_result.csv')
        label = d.loc[d.loc[:,'file']==file_list[i],'label']
        predict = d.loc[d.loc[:,'file']==file_list[i],'predict']

        if label.iloc[0] == 0:
            comment1 = 'None'
        if label.iloc[0] == 1:
            comment1 = 'Need co-worker'
        if label.iloc[0] == 2:
            comment1 = 'Need safetybelt'
        if label.iloc[0] == 3:
            comment1 = 'Need safetybelt, co-worker'
        if label.iloc[0] == 4:
            comment1 = 'Need safetybelt, helmet'
        if label.iloc[0] == 5:
            comment1 = 'Need safetybelt, helmet, co-worker'

        if predict.iloc[0] == 0:
            comment2 = 'None'
        if predict.iloc[0] == 1:
            comment2 = 'Need co-worker'
        if predict.iloc[0] == 2:
            comment2 = 'Need safetybelt'
        if predict.iloc[0] == 3:
            comment2 = 'Need safetybelt, co-worker'
        if predict.iloc[0] == 4:
            comment2 = 'Need safetybelt, helmet'
        if predict.iloc[0] == 5:
            comment2 = 'Need safetybelt, helmet, co-worker'
         
        if comment1 == comment2 :
            cv2.imwrite(f"./cam/{comment1}_{file_list[i]}.png", np_img)
            cv2.imwrite(f"./cam/{comment1}_{file_list[i]}_result.png", mask)