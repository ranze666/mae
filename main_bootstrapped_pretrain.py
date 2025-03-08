# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae
import model_bootstrapped_mae

from engine_pretrain import train_one_epoch
import main_finetune

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_tiny_patch4', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=5e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=80, metavar='N',
                        help='epochs to warmup LR')
    # parser.add_argument('--update_epoch_list',type=list,default=[0,5,10,20,40,80,160,200])

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir/bootmae',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir/bootmae',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    parser.add_argument('--ema_decay_init',default=0.7)
    parser.add_argument('--ema_decay_final',default=0.9999)
    parser.add_argument('--ema_decay_warmup_epoch',default=80)



    return parser


finetune_eval_dict = {
    "batch_size": 256,
    "epochs": 1,
    "accum_iter": 4,
    
    # Model parameters
    "model": "vit_tiny_patch4",
    "input_size": 32,
    "drop_path": 0.1,

    # Optimizer parameters
    "clip_grad": None,
    "weight_decay": 0.05,
    "lr": None,
    "blr": 5e-4,
    "layer_decay": 0.65,
    "min_lr": 1e-6,
    "warmup_epochs": 40,

    # Augmentation parameters
    "color_jitter": None,
    "aa": "rand-m9-mstd0.5-inc1",
    "smoothing": 0.1,

    # Random Erase params
    "reprob": 0.25,
    "remode": "pixel",
    "recount": 1,
    "resplit": False,

    # Mixup params
    "mixup": 0.8,
    "cutmix": 1.0,
    "cutmix_minmax": None,
    "mixup_prob": 1.0,
    "mixup_switch_prob": 0.5,
    "mixup_mode": "batch",

    # Finetuning params
    "finetune": "",
    "global_pool": True,
    "cls_token": False,

    # Dataset parameters
    "data_path": "",
    "nb_classes": 10,
    "output_dir": "./output_dir/tmp",
    "log_dir": "./output_dir/tmp",
    "device": "cuda",
    "seed": 0,
    "resume": "",

    "start_epoch": 0,
    "eval": False,
    "dist_eval": False,
    "num_workers": 10,
    "pin_mem": True,

    # Distributed training parameters
    "world_size": 1,
    "local_rank": -1,
    "dist_on_itp": False,
    "dist_url": "env://"
}
from types import SimpleNamespace
finetune_eval_namespace = SimpleNamespace(**finetune_eval_dict)

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.6, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # cifar10数据集
    dataset_train = datasets.CIFAR10(root="./data", train=True,
                                           transform=transform_train, download=True)
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    # print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = model_bootstrapped_mae.BootstrappedMAE(args)

    # pixel_pretrain_checkpoint = torch.load('./output_dir/mae/checkpoint-40.pth', map_location="cpu")
    # encoder_state_dict = {k: v for k, v in pixel_pretrain_checkpoint["model"].items() if "decoder" not in k}
    # model.student_model.load_state_dict(encoder_state_dict,strict=False)
    # model.teacher_model.load_state_dict(encoder_state_dict,strict=False)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp.student_model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # if epoch in args.update_epoch_list:
        #     model.update_teacher()

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        model.update_decay_cosine(epoch,args)
        model.update_teacher(method='ema')
        log_writer.add_scalar('ema_decay', model.ema_decay, epoch)
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(    # 只保存student_model
                args=args, model=model, model_without_ddp=model_without_ddp.student_model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            
            finetune_eval_namespace.finetune = f'{args.output_dir}/checkpoint-{epoch}.pth'
            acc = main_finetune.main(finetune_eval_namespace)
            print('finetune acc:', acc)

            log_writer.add_scalar('finetune_first_epoch_acc', acc ,epoch)



        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, f"run_{timestamp}")
        os.makedirs(args.log_dir, exist_ok=True)
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
        os.makedirs(args.output_dir, exist_ok=True)

    # 保存args方便后续比较
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, "args.txt")  
    with open(args_path, "w") as f:
        json.dump(args_dict, f, indent=4)


    main(args)
