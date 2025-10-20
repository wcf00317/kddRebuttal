# wcf00317/alrl/alrl-reward_model/run_rl_with_pseudolabel_reward.py

import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
import datetime

import torch
import torch.nn as nn
import yaml
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader, ConcatDataset, TensorDataset

# --- 复用您项目中的模块 ---
from models.model_utils import create_models, get_video_candidates, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward

cudnn.benchmark = False
cudnn.deterministic = True


def calculate_pseudo_label_reward(net, selected_batch_indices, train_set, val_loader, criterion, base_acc, args):
    """
    通过伪标签来估算奖励，避免“偷窥”真实标签。
    这是一个更公平的、受 MGRAL 启发的奖励计算方法。
    """
    print("正在通过伪标签估算奖励 (MGRAL-style Fair △ACC)...")

    # 1. 创建一个临时的模型副本和优化器
    net_copy = deepcopy(net)
    optimizer_temp = torch.optim.SGD(net_copy.parameters(), lr=args.lr)

    # 2. 为选中的批次生成伪标签
    pseudo_labels = []
    net.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        for vid_idx in selected_batch_indices:
            # 使用 get_video 获取用于评估的、标准的中心裁剪视频
            video_clip = train_set.get_video(vid_idx).cuda().unsqueeze(0)  # Shape: [1, C, T, H, W]
            # print(video_clip.shape)
            outputs = net(video_clip, return_loss=False)
            outputs = net.cls_head(outputs)  # Shape: [1, num_classes]
            pseudo_label = outputs.argmax(dim=1).item()
            pseudo_labels.append(pseudo_label)

    print(f"  - 为 {len(selected_batch_indices)} 个样本生成了伪标签。")

    # 3. 创建一个临时的混合数据集
    # a. 已有标注的数据集
    labeled_subset = Subset(train_set, list(train_set.labeled_video_ids))
    # b. 带有伪标签的新数据集
    pseudo_labeled_subset = Subset(train_set, selected_batch_indices)
    # 重写标签
    pseudo_labeled_subset.dataset = deepcopy(pseudo_labeled_subset.dataset)
    for i, original_idx in enumerate(selected_batch_indices):
        # 找到 video_list_info 中对应的条目并修改标签
        for j, info in enumerate(pseudo_labeled_subset.dataset.video_list_info):
            if info[2] == original_idx:  # original_index is at position 2
                # (video_name, label, original_index)
                pseudo_labeled_subset.dataset.video_list_info[j] = (info[0], str(pseudo_labels[i]), info[2])
                break

    # c. 合并两个数据集
    combined_dataset = ConcatDataset([labeled_subset, pseudo_labeled_subset])
    temp_loader = DataLoader(combined_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

    # --- 4. 在这个混合了真实标签和伪标签的数据集上进行微调 ---
    _, new_estimated_acc = train_har_for_reward(net_copy, temp_loader, val_loader, optimizer_temp, criterion, args)

    reward = new_estimated_acc - base_acc

    # 释放副本占用的内存
    del net_copy, optimizer_temp, temp_loader, combined_dataset, pseudo_labeled_subset, labeled_subset
    torch.cuda.empty_cache()

    return reward, new_estimated_acc


def main(args):
    # --- 1. 初始化和配置加载 (不变) ---
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

    # --- 2. 创建模型、数据、优化器 (不变) ---
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
        print('--- 启动 RL 训练 (奖励信号: 伪标签估算 △ACC) ---')

        # --- 3. RL 相关变量初始化 (不变) ---
        Transition = namedtuple('Transition', (
        'state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
        memory = ReplayMemory(args.rl_buffer)
        TARGET_UPDATE = 5
        steps_done = 0
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

        # 初始化基准准确率
        print("正在计算初始模型的基准准确率...")
        initial_loader = DataLoader(Subset(train_set, list(train_set.labeled_video_ids)),
                                    batch_size=args.train_batch_size, shuffle=True,
                                    num_workers=args.workers)
        initial_optimizer = create_and_load_optimizers(
            net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
            momentum=args.momentum, ckpt_path=args.ckpt_path,
            exp_name_toload=None, exp_name=args.exp_name,
            snapshot=args.snapshot, checkpointer=False, load_opt=False
        )[0]
        _, past_val_acc = train_har_for_reward(net, initial_loader, val_loader, initial_optimizer, criterion, args)
        print(f"初始基准准确率: {past_val_acc:.4f}")

        num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter

        # --- 4. 主动学习与 RL 训练循环 ---
        for i in range(num_al_steps):
            print(f'\\n--------------- RL (Pseudo-Label Reward) 回合 {i + 1}/{num_al_steps} ---------------')

            current_state, candidate_indices, _ = compute_state_for_har(
                args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
            )
            action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
            actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

            # --- 核心修改: 使用新的伪标签奖励函数 ---
            estimated_reward, _ = calculate_pseudo_label_reward(
                net, actual_video_ids_to_label, train_set, val_loader, criterion, past_val_acc, args
            )
            print(f"估算奖励 (伪标签 △ACC): {estimated_reward:.4f}")

            # (后续流程保持不变)
            add_labeled_videos(args, [], actual_video_ids_to_label, train_set,
                               budget=args.budget_labels, n_ep=i)

            current_labeled_indices = list(train_set.labeled_video_ids)
            train_loader = DataLoader(Subset(train_set, current_labeled_indices),
                                      batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=False)

            print('使用新选择的视频更新主HAR网络...')
            _, past_val_acc = train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args)
            print(f"主模型已更新，新的基准准确率: {past_val_acc:.4f}")

            next_state = None
            if train_set.get_num_labeled_videos() < args.budget_labels:
                next_state, _, _ = compute_state_for_har(
                    args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
                )

            reward_tensor = torch.tensor([estimated_reward], dtype=torch.float, device='cuda')
            memory.push(current_state, action, next_state, reward_tensor)

            if len(memory) >= args.dqn_bs:
                optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                    BATCH_SIZE=args.dqn_bs)

            if i % TARGET_UPDATE == 0:
                print('更新目标网络...')
                target_net.load_state_dict(policy_net.state_dict())

        # --- 5. 最终收敛训练 (不变) ---
        print("\\n预算已用尽。在所有已选数据上训练HAR模型至收敛...")
        logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                             log_name='final_convergence_log.txt')
        final_labeled_indices = list(train_set.labeled_video_ids)
        final_train_loader = DataLoader(Subset(train_set, final_labeled_indices), batch_size=args.train_batch_size,
                                        shuffle=True,
                                        num_workers=args.workers, drop_last=False)
        _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net,
                                                criterion, optimizer, val_loader,
                                                best_record, logger, scheduler,
                                                schedulerP, final_train=True)
        print(f"收敛后的最终验证集准确率: {final_val_acc:.4f}")
        torch.save(policy_net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'policy_final.pth'))


if __name__ == '__main__':
    args = parser.get_arguments()
    main(args)