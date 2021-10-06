import os
import gc
import time
import wandb
import logging 

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms 
from torch.cuda.amp import GradScaler, autocast

from model.dataset import CustomDataset
from model.ViT import ViT
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, LambdaLR, CosineAnnealingLR

from util import TqdmLoggingHandler, write_log, label_smoothing_loss

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train_epoch(args, epoch, model, dataloader, optimizer, scheduler, scaler, logger, device):

    start_time_e = time.time()
    model = model.train()
    for i, (img, label) in enumerate(dataloader):

        # Optimizer
        optimizer.zero_grad()

        # Input, output
        x = img.to(device, non_blocking=True)
        y = label.to(device, non_blocking=True)

        logit = model(x)
        
        #loss
        loss = label_smoothing_loss(logit, y, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        acc = (((logit.argmax(dim=1) == y).sum()) / y.size(0)) * 100

        if i == 0 or freq == args.print_freq or i == (len(dataloader)-1):
            batch_log = "[Epoch:%d][%d/%d] train_loss:%2.3f  | train_acc:%02.2f | learning_rate:%3.6f | spend_time:%3.2fmin" \
                    % (epoch+1, i, len(dataloader), 
                    loss.item(), acc.item(), optimizer.param_groups[0]['lr'], 
                    (time.time() - start_time_e) / 60)
            write_log(logger, batch_log)
            freq = 0
        freq += 1

def valid_epoch(args, model, dataloader, device):

    # Validation setting
    model = model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(dataloader):

            # Input, output setting
            x = img.to(device, non_blocking=True)
            y = label.to(device, non_blocking=True)

            # Model
            logit = model(x)
            first_token = logit
            # first_token = logit

            # Loss calculate
            loss = F.cross_entropy(first_token, y)

            # Print loss value only training
            acc = (((first_token.argmax(dim=1) == y).sum()) / y.size(0)) * 100
            val_loss += loss.item()
            val_acc += acc.item()

    return val_loss, val_acc

def training(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # data loader setting

    write_log(logger, 'Load data...')
    gc.disable()
    transform_dict = {
        'train' : transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ColorJitter(brightness=(0.5,2)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'valid' : transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }

    dataset_dict = {
        'train' : CustomDataset(data_path = args.data_path, transform = transform_dict['train'], phase = 'train'),
        'valid' : CustomDataset(data_path = args.data_path, transform = transform_dict['valid'], phase = 'valid')
    }

    dataloader_dict = {
        'train' : DataLoader(dataset_dict['train'], drop_last = True, batch_size = args.batch_size, shuffle=True,
        pin_memory = True),
        'valid' : DataLoader(dataset_dict['valid'], drop_last = False, batch_size = args.batch_size, shuffle=False,
        pin_memory = False)
    }
    
    gc.enable()

    write_log(logger, f"Total number of trainingsets  iterations - {len(dataset_dict['train'])}, {len(dataloader_dict['train'])}")


    # Model

    write_log(logger, "Instantiating models...")
    model = ViT( 
        in_channels = args.in_channels, 
        patch_size = args.patch_size, 
        emb_size = args.emb_size, 
        img_size = args.img_size, 
        depth = args.depth, 
        n_classes = args.n_classes
        ).to(device)

    # Optimizer
    optimizer = optimizer = Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr, eps=1e-8)
    scheduler = scheduler = scheduler = StepLR(optimizer, step_size=len(dataloader_dict['train']), gamma=1)
    scaler = GradScaler()


    # model resume

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.model_path, 'checkpoint.pth.tar'), map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        model = model.train()
        model = model.to(device)
        del checkpoint

    ### train

    best_val_acc = 0

    write_log(logger, 'Train')

    for epoch in range(start_epoch, args.num_epochs):
        
        train_epoch(args, epoch, model, dataloader_dict['train'], optimizer, scheduler, scaler, logger, device)
        val_loss, val_acc = valid_epoch(args, model, dataloader_dict['valid'], device)

        val_loss /= len(dataloader_dict['valid'])
        val_acc /= len(dataloader_dict['valid'])
        write_log(logger, 'Validation Loss: %3.3f' % val_loss)
        write_log(logger, 'Validation Accuracy: %3.2f%%' % val_acc)
        if val_acc > best_val_acc:
            write_log(logger, 'Checkpoint saving...')
            # Checkpoint path setting
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            # Save
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict()
            }, os.path.join(args.model_path, f'checkpoint.pth.tar'))
            best_val_acc = val_acc
            best_epoch = epoch
        else:
            else_log = f'Still {best_epoch} epoch accuracy({round(best_val_acc, 2)})% is better...'
            write_log(logger, else_log)

    # 3)
    write_log(logger, f'Best Epoch: {best_epoch}')
    write_log(logger, f'Best Accuracy: {round(best_val_acc, 2)}')