# -*- coding: utf-8 -*-
# 文件名: run_rl_with_sabotaged_reward.py
#
# “使坏”版：复现MGRAL基线，并通过颠倒奖励信号来主动破坏学习稳定性。
# 奖励 = 被破坏的真实 ΔACC:
#   - 当模型取得正向进展 (ΔACC > 0) 时，奖励会被乘以一个[-1, 0.1]的随机因子，从而惩罚或无视成功。
#   - 当模型表现变差 (ΔACC <= 0) 时，负面奖励保持不变。
#   - 结合了动态验证集和非确定性cuDNN，以最大化奖励的混乱和不可预测性。

import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn as nn
import yaml
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Subset, DataLoader

# --- 复用项目模块 ---
from models.model_utils import create_models, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward

# 增加随机性
cudnn.benchmark = True
cudnn.deterministic = False


def main(args):
    # --- 配置加载与初始化 ---
    if getattr(args, 'config', None):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
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
    shutil.copy(sys.argv[0], os.path.join(args.ckpt_path, args.exp_name, sys.argv[0].rsplit('/', 1)[-1]))

    # --- 创建模型、数据、优化器 ---
    net, policy_net, target_net = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path, num_classes=args.num_classes,
                                                use_policy=True, embed_dim=args.embed_dim)

    _, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, model_type=args.model_type, n_workers=args.workers, clip_len=args.clip_len
    )
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer, optimizerP = create_and_load_optimizers(
        net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
        momentum=args.momentum, ckpt_path=args.ckpt_path,
        exp_name_toload=args.exp_name_toload, exp_name=args.exp_name,
        snapshot=args.snapshot, checkpointer=args.checkpointer,
        load_opt=args.load_opt, policy_net=policy_net, lr_dqn=args.lr_dqn
    )

    if args.train:
        print('--- 启动 RL 训练 (奖励信号: 被“使坏”的真实 ΔACC) ---')

        Transition = namedtuple('Transition', (
            'state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
        memory = ReplayMemory(args.rl_buffer)
        TARGET_UPDATE = 5
        steps_done = 0
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

        # 动态计算基准准确率
        print("正在计算初始模型的基准准确率 (base_acc)...")
        initial_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)),
                                    batch_size=args.train_batch_size, shuffle=True,
                                    num_workers=args.workers)

        # 为了获取初始准确率，我们先动态抽样一次验证集
        val_indices = list(range(len(val_loader.dataset)))
        random.shuffle(val_indices)
        subset_size = int(len(val_indices) * 0.20)
        small_val_indices = val_indices[:subset_size]
        small_val_subset = Subset(val_loader.dataset, small_val_indices)
        initial_small_val_loader = DataLoader(small_val_subset, batch_size=args.val_batch_size, shuffle=False,
                                              num_workers=args.workers)

        initial_optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
        _, past_val_acc = train_har_for_reward(net, initial_loader, initial_small_val_loader, initial_optimizer,
                                               criterion, args)
        print(f"初始基准准确率 (past_val_acc): {past_val_acc:.4f}")

        num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter

        for i in range(num_al_steps):
            print(f'\n--------------- RL (被“使坏”的 ΔACC Reward) 回合 {i + 1}/{num_al_steps} ---------------')

            # 每轮都重新抽样验证集
            val_indices = list(range(len(val_loader.dataset)))
            random.shuffle(val_indices)
            subset_size = int(len(val_indices) * 0.20)
            small_val_indices = val_indices[:subset_size]
            small_val_subset = Subset(val_loader.dataset, small_val_indices)
            small_val_loader = DataLoader(small_val_subset, batch_size=args.val_batch_size, shuffle=False,
                                          num_workers=args.workers)
            print(f"已动态抽样新的验证子集，大小: {len(small_val_loader.dataset)}")

            current_state, candidate_indices, _ = compute_state_for_har(
                args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
            )

            action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
            actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

            print("正在计算真实奖励 (ΔACC)...")
            net_copy_for_reward = deepcopy(net)
            optimizer_for_reward = torch.optim.SGD(net_copy_for_reward.parameters(), lr=args.lr)
            temp_set_for_reward = deepcopy(train_set)

            add_labeled_videos(args, [], actual_video_ids_to_label, temp_set_for_reward, budget=args.budget_labels,
                               n_ep=i)

            temp_loader_for_reward = DataLoader(
                Subset(temp_set_for_reward, list(temp_set_for_reward.labeled_video_ids)),
                batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

            _, new_val_acc = train_har_for_reward(net_copy_for_reward, temp_loader_for_reward, small_val_loader,
                                                  optimizer_for_reward, criterion, args)

            real_reward = new_val_acc - past_val_acc

            # ==========================================================
            # ✅ 修改点: 对模型“使坏”，颠倒奖励信号
            # ==========================================================
            if real_reward > 0:
                sabotage_factor = np.random.uniform(-1, 0.1)
                sabotaged_reward = real_reward * sabotage_factor
                print(f"“使坏”启动！正向奖励 {real_reward:.6f} 被乘以 {sabotage_factor:.4f}，变为 {sabotaged_reward:.6f}")
                real_reward = sabotaged_reward
            else:
                print(f"真实奖励 (ΔACC): {real_reward:.6f} (new: {new_val_acc:.6f} vs past: {past_val_acc:.6f})")

            del net_copy_for_reward, optimizer_for_reward, temp_set_for_reward, temp_loader_for_reward
            torch.cuda.empty_cache()

            add_labeled_videos(args, [], actual_video_ids_to_label, train_set,
                               budget=args.budget_labels, n_ep=i)

            current_labeled_indices = list(train_set.labeled_video_ids)
            train_loader = DataLoader(Subset(train_set, current_labeled_indices),
                                      batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=False)

            print('使用新选择的视频更新主HAR网络...')
            _, past_val_acc = train_har_for_reward(net, train_loader, small_val_loader, optimizer, criterion, args)
            print(f"主模型已更新, 新的基准准确率 (past_val_acc): {past_val_acc:.4f}")

            next_state = None
            if train_set.get_num_labeled_videos() < args.budget_labels:
                next_state, _, _ = compute_state_for_har(
                    args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
                )

            reward_tensor = torch.tensor([real_reward], dtype=torch.float, device='cuda')
            memory.push(current_state, action, next_state, reward_tensor)

            if len(memory) >= args.dqn_bs:
                optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP,
                                    GAMMA=args.dqn_gamma, BATCH_SIZE=args.dqn_bs)

            if i % TARGET_UPDATE == 0:
                print('更新目标网络...')
                target_net.load_state_dict(policy_net.state_dict())

        # --- 最终收敛训练 ---
        print("\n预算已用尽。在所有已选数据上训练HAR模型至收敛...")
        logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                             log_name='final_convergence_log.txt')
        final_labeled_indices = list(train_set.labeled_video_ids)
        final_train_loader = DataLoader(Subset(train_set, final_labeled_indices), batch_size=args.train_batch_size,
                                        shuffle=True, num_workers=args.workers, drop_last=False)
        # 最终评估时，使用完整的验证集
        _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net,
                                                criterion, optimizer, val_loader,
                                                best_record, logger, scheduler,
                                                schedulerP, final_train=True)
        print(f"收敛后的最终验证集准确率: {final_val_acc:.4f}")
        torch.save(policy_net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'policy_final.pth'))


if __name__ == '__main__':
    args = parser.get_arguments()
    main(args)