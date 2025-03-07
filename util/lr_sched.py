# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate with half-cycle cosine after warmup"""
#     if epoch < args.warmup_epochs:
#         lr = args.lr * epoch / args.warmup_epochs 
#     else:
#         lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
#             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
#     for param_group in optimizer.param_groups:
#         if "lr_scale" in param_group:
#             param_group["lr"] = lr * param_group["lr_scale"]
#         else:
#             param_group["lr"] = lr
#     return lr


def adjust_learning_rate(optimizer, epoch, args):
    
    # """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        global_lr = args.lr * epoch / args.warmup_epochs 
    else:
        global_lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    # global_lr = args.lr

    if hasattr(args, 'update_epoch_list'):
        # 局部warmup
        for i in range(len(args.update_epoch_list) - 1):
            start, end = args.update_epoch_list[i], args.update_epoch_list[i + 1]
            if start <= epoch < end:  
                stage_progress = (epoch - start) / (end - start)  # 局部 warmup 进度
                warmup_ratio = 0.4
                if stage_progress < warmup_ratio:
                    lr = global_lr * stage_progress / warmup_ratio
                else:
                    lr = args.min_lr + (global_lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (stage_progress-warmup_ratio) / (1-warmup_ratio)))

                break
    else:
        lr = global_lr


    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr