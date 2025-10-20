# 文件名: run_random_incremental_milestones_v3.py

import os
import sys
import shutil
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from copy import deepcopy
import math

# 从您的项目中导入必要的模块
from models.model_utils import create_models, add_labeled_videos
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
import utils.parser as parser
from torch.utils.data import Subset, DataLoader

# 确保可以从您的项目中导入 train_har_classifier 函数
from run_rl_with_alrm import train_har_classifier

cudnn.benchmark = False
cudnn.deterministic = True


def main(args):
    # --- 1. 初始化和配置加载 ---
    if getattr(args, 'config', None):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    arg_key = f"{key}_{sub_key}"
                    if not hasattr(args, arg_key) or getattr(args, arg_key) is None:
                        setattr(args, arg_key, sub_value)
            else:
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    parser.save_arguments(args)

    # --- 2. 创建模型、加载数据、设置日志 ---
    net, _, _ = create_models(dataset=args.dataset,
                              model_cfg_path=args.model_cfg_path,
                              model_ckpt_path=args.model_ckpt_path,
                              num_classes=args.num_classes,
                              use_policy=False,
                              embed_dim=args.embed_dim)
    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    _, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        n_workers=args.workers, clip_len=args.clip_len, model_type=args.model_type, dataset_name=args.dataset
    )
    total_videos = len(train_set)

    milestone_logger, _, _ = get_logfile(args.ckpt_path, args.exp_name, checkpointer=False, snapshot=None,
                                         log_name='random_milestone_log.txt')
    milestone_logger.set_names(['Data_Ratio (%)', 'Labeled_Count', 'Validation_Accuracy'])
    accuracy_records = []

    # --- 4. 初始训练阶段 ---
    initial_labeled_indices = list(train_set.labeled_video_ids)
    initial_train_subset = Subset(train_set, initial_labeled_indices)
    initial_train_loader = DataLoader(initial_train_subset, batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=False)
    print(f"--- 开始初始训练阶段, 当前样本数: {len(initial_train_subset)} ---")

    # 为初始训练创建优化器和调度器
    optimizer = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr,
                                           wd=args.weight_decay, momentum=args.momentum, ckpt_path=args.ckpt_path,
                                           exp_name_toload=None, exp_name=args.exp_name, snapshot=None,
                                           checkpointer=False, load_opt=False)[0]
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    best_record = {'top1_acc': 0.0}

    _, initial_val_acc = train_har_classifier(args, 0, initial_train_loader, net,
                                              criterion, optimizer, val_loader, best_record,
                                              logger=None, scheduler=scheduler, schedulerP=None, final_train=True)

    num_initial_labeled = len(initial_labeled_indices)
    initial_ratio = num_initial_labeled / total_videos * 100
    print(f"初始训练完成！已标注 {num_initial_labeled} 个样本 ({initial_ratio:.1f}%), 验证集准确率: {initial_val_acc:.4f}")

    # --- 5. 增量主动学习循环与修正后的里程碑记录 ---
    milestone_ratios = np.arange(0.05, 0.51, 0.05)
    milestone_counts = [int(total_videos * r) for r in milestone_ratios]

    next_milestone_idx = 0
    while next_milestone_idx < len(milestone_counts) and num_initial_labeled >= milestone_counts[next_milestone_idx]:
        ratio_percent = milestone_ratios[next_milestone_idx] * 100
        print(f"初始数据已达到或超过里程碑: {ratio_percent:.0f}%")
        accuracy_records.append({'ratio': ratio_percent, 'count': num_initial_labeled, 'accuracy': initial_val_acc})
        milestone_logger.append([ratio_percent, num_initial_labeled, initial_val_acc])
        next_milestone_idx += 1

    max_budget = int(total_videos * 0.5)

    if train_set.get_num_labeled_videos() >= max_budget:
        num_al_steps = 0
    else:
        num_al_steps = math.ceil((max_budget - train_set.get_num_labeled_videos()) / args.num_each_iter)

    print(f"\n--- 开始随机增量学习循环，共 {num_al_steps} 轮，直到达到 50% 的数据量 ---")

    for i in range(num_al_steps):
        num_labeled_before = train_set.get_num_labeled_videos()
        print(f'\n----- 随机选择第 {i + 1}/{num_al_steps} 轮: 当前已标注 {num_labeled_before}/{max_budget} -----')

        unlabeled_indices = train_set.get_candidates_video_ids()

        num_to_select = min(args.num_each_iter, len(unlabeled_indices))
        if num_labeled_before + num_to_select > max_budget:
            num_to_select = max_budget - num_labeled_before

        if num_to_select <= 0:
            print("已达到或超过50%预算，停止选择。")
            break

        selected_indices = random.sample(unlabeled_indices, num_to_select)

        print(f"随机选择了 {len(selected_indices)} 个视频。")
        add_labeled_videos(args, [], selected_indices, train_set, budget=max_budget, n_ep=i)

        print('在扩充后的数据集上微调 HAR 网络...')
        current_labeled_indices = list(train_set.labeled_video_ids)
        train_subset = Subset(train_set, current_labeled_indices)
        current_train_loader = DataLoader(train_subset, batch_size=args.train_batch_size, shuffle=True,
                                          num_workers=args.workers, drop_last=False)

        # ======================================================================
        # 核心修正 1: 重新创建 Optimizer 和 Scheduler，重置学习率
        # ======================================================================
        print("为当前微调阶段重置优化器和学习率调度器...")
        optimizer = create_and_load_optimizers(net=net, opt_choice=args.optimizer, lr=args.lr,
                                               wd=args.weight_decay, momentum=args.momentum, ckpt_path=args.ckpt_path,
                                               exp_name_toload=None, exp_name=args.exp_name, snapshot=None,
                                               checkpointer=False, load_opt=False)[0]
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)

        # ======================================================================
        # 核心修正 2: 重置 best_record，使其只记录当前阶段的最佳表现
        # ======================================================================
        best_record = {'top1_acc': 0.0}

        _, val_acc_after_finetune = train_har_classifier(args, 0, current_train_loader, net,
                                                         criterion, optimizer, val_loader, best_record,
                                                         logger=None, scheduler=scheduler, schedulerP=None,
                                                         final_train=True, epochs_to_run=args.al_train_epochs)

        num_labeled_after = train_set.get_num_labeled_videos()
        print(f"微调后，已标注 {num_labeled_after} 个样本, 验证集准确率为: {val_acc_after_finetune:.4f}")

        while next_milestone_idx < len(milestone_counts) and num_labeled_after >= milestone_counts[next_milestone_idx]:
            ratio_percent = milestone_ratios[next_milestone_idx] * 100
            print(f"--- 达到新的里程碑: {ratio_percent:.0f}% ---")

            # 使用微调后的最终准确率作为这个里程碑的记录
            accuracy_records.append(
                {'ratio': ratio_percent, 'count': num_labeled_after, 'accuracy': val_acc_after_finetune})
            milestone_logger.append([ratio_percent, num_labeled_after, val_acc_after_finetune])

            next_milestone_idx += 1

    # --- 6. 结束并打印最终结果 ---
    milestone_logger.close()
    print("\n" + "=" * 50)
    print("       随机采样增量评估 (里程碑模式) 完成！")
    print("=" * 50)
    print(f"{'Data Ratio (%)':<15}{'Sample Count':<15}{'Validation Accuracy':<20}")
    print("-" * 50)
    for record in accuracy_records:
        print(f"{record['ratio']:<14.0f}%{record['count']:<15}{record['accuracy']:.4f}")
    print("-" * 50)
    print(f"详细日志已保存至: {os.path.join(args.ckpt_path, args.exp_name, 'random_milestone_log.txt')}")


if __name__ == '__main__':
    args = parser.get_arguments()
    main(args)