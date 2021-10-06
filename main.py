import time
import argparse
import numpy as np
import random
import torch

from train import training
from test import test
from vit_explain import visual
# from test

SEED = 940103

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def main(args):
    
    total_start_time = time.time()
    
    if args.isTrain:
        training(args)
    if args.isTest:
        test(args)
    if args.isVisual:
        visual(args)
    # if args.testing:
        

    print(f'{round((time.time()-total_start_time)/60, 3)}min spend')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parsing Method')

    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--isTrain',action='store_true')
    parser.add_argument('--isTest',action='store_true')
    parser.add_argument('--isVisual',action='store_true')

    parser.add_argument('--data_path', default='./data', type=str)
    parser.add_argument('--model_path', default='./checkpoints', type=str)
    parser.add_argument('--cam_path', default='./cam', type=str)

    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--emb_size', default=768, type=int)
    parser.add_argument('--n_classes', default=6, type=int)
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--depth', default=12, type=int)

    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--clip_grad_norm', default=5, type=int)

    parser.add_argument('--print_freq', default=20, type=int)
    args = parser.parse_args()

    main(args)