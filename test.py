import os
import gc
import time
import wandb
import logging 
import json
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms 
from torch.cuda.amp import GradScaler, autocast
from torchsummary import summary

from model.dataset import CustomDataset
from model.ViT import ViT
from util import TqdmLoggingHandler, write_log, label_smoothing_loss


def test(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    write_log(logger, "Load Test data...")
    gc.disable()
    transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    )
    
    dataset_test = CustomDataset(data_path=args.data_path,
                                 transform=transform_test, 
                                 phase='test')
    dataloader_test = DataLoader(dataset_test, drop_last=False,
                                 batch_size=args.batch_size, shuffle=True, pin_memory=True
                                 )
    gc.enable()

    write_log(logger, f"Total number of testingsets iterations - {len(dataset_test)}, {len(dataloader_test)}")

    write_log(logger, "Loading models...")

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


    for name, p in model.named_parameters():
        print("###############",name,"#################")
    predicted_list = list()
    label_list = list()
    file_list = list()
    test_acc = 0

    start_time = time.time()

    write_log(logger, "Test start!")

    with torch.no_grad():
        for print_step, (img, label, file) in enumerate(dataloader_test):

            x = img.to(device, non_blocking=True)

            y_list = label.tolist()
            label_list.extend(y_list)
            y = label.to(device, non_blocking=True)

            file_list.extend(file)

            logit = model(x)
            acc = (((logit.argmax(dim=1) == y).sum()) / y.size(0)) * 100
            test_acc += acc.item()
            
            predict = logit.argmax(dim=1).tolist()
            predicted_list.extend(predict)

    test_acc /= len(dataloader_test)

    # Print progress
    if print_step % args.print_freq == 0:
        write_log(logger, f'[{print_step}/{len(dataloader_test)}] spend_time: {round((time.time() - start_time) / 60, 3)}min')
     
    result = {
        'test_acc' : test_acc,
        'file_list' : file_list,
        'label' : label_list,
        'predict' : predicted_list
    }
    print(test_acc)
    with open(f'{args.model_path}/result.json','w') as outfile:
        json.dump(result, outfile)
