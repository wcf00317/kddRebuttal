# 文件名: train_Acc_incremental.py
# (基于 run_acc_reward.py 的 RL 策略 和 run_random_incremental.py 的增量评估框架)
# (简体中文版，目标 5% - 35%)

import os
import sys
import shutil
import random
import numpy as np
from collections import namedtuple
from copy import deepcopy
import datetime
import yaml
import math

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader

# --- 导入您项目中的所有必要函数 ---
from models.model_utils import create_models, get_video_candidates, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser

# 从 run_rl_with_alrm.py 导入训练函数 (假设 run_rl_with_alrm.py 在可导入的路径中)
# 我们需要这两个函数：
# 1. train_har_classifier: 用于在里程碑处训练至收敛并评估
# 2. train_har_for_reward: 用于计算奖励时的短时微调
try:
    from run_rl_with_alrm import train_har_classifier, train_har_for_reward
except ImportError:
    print("错误: 无法从 'run_rl_with_alrm' 导入 'train_har_classifier' 或 'train_har_for_reward'。")
    print("请确保 run_rl_with_alrm.py 在您的 PYTHONPATH 中。")
    sys.exit(1)

cudnn.benchmark = False
cudnn.deterministic = True


def main(args):
    # --- 1. 初始化和配置加载 ---
    # (来自 run_random_incremental.py)
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
    # 保存当前脚本的副本 (来自 run_acc_reward.py)
    try:
        shutil.copy(sys.argv[0], os.path.join(args.ckpt_path, args.exp_name, sys.argv[0].rsplit('/', 1)[-1]))
    except Exception as e:
        print(f"警告：无法复制脚本文件。错误: {e}")

    # --- 2. 创建模型、数据、优化器 ---
    # (来自 run_acc_reward.py，因为我们需要 Policy Net)
    net, policy_net, target_net = create_models(dataset=args.dataset,
                                                model_cfg_path=args.model_cfg_path,
                                                model_ckpt_path=args.model_ckpt_path,
                                                num_classes=args.num_classes,
                                                use_policy=True,
                                                embed_dim=args.embed_dim)
    net.cuda()
    policy_net.cuda()
    target_net.cuda()

    _, train_set, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        n_workers=args.workers, clip_len=args.clip_len, model_type=args.model_type, dataset_name=args.dataset
    )
    total_videos = len(train_set)
    criterion = nn.CrossEntropyLoss().cuda()

    # (来自 run_acc_reward.py)
    optimizer, optimizerP = create_and_load_optimizers(
        net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
        momentum=args.momentum, ckpt_path=args.ckpt_path,
        exp_name_toload=args.exp_name_toload, exp_name=args.exp_name,
        snapshot=args.snapshot, checkpointer=args.checkpointer,
        load_opt=args.load_opt, policy_net=policy_net, lr_dqn=args.lr_dqn
    )

    # --- 3. RL 相关变量初始化 ---
    # (来自 run_acc_reward.py)
    Transition = namedtuple('Transition', (
        'state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
    memory = ReplayMemory(args.rl_buffer)
    TARGET_UPDATE = args.target_update if hasattr(args, 'target_update') else 5  # 从 args 获取或设为默认值
    steps_done = 0
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    # 注意：我们将在增量循环内部管理 HAR 分类器 (net) 的优化器和调度器

    # --- 4. 增量日志和里程碑设置 ---
    # (来自 run_random_incremental.py)
    log_file_name = f'acc_reward_incremental_log_seed{args.seed}.txt'
    milestone_logger, _, _ = get_logfile(args.ckpt_path, args.exp_name, checkpointer=False, snapshot=None,
                                         log_name=log_file_name)
    milestone_logger.set_names(['Data_Ratio (%)', 'Labeled_Count', 'Validation_Accuracy'])
    accuracy_records = []

    # ======================================================================
    # 关键修改：根据用户请求，将里程碑设置为 5% 到 35%
    milestone_ratios = np.arange(0.05, 0.36, 0.05)  # [0.05, 0.1, ..., 0.35]
    # ======================================================================

    milestone_counts = [int(total_videos * r) for r in milestone_ratios]
    max_budget = milestone_counts[-1]  # 最终预算自动设为 35%

    # --- 5. 初始训练阶段 (在 5% 数据上) ---
    # (改编自 run_random_incremental.py)
    initial_labeled_indices = list(train_set.labeled_video_ids)
    num_initial_labeled = len(initial_labeled_indices)
    initial_train_subset = Subset(train_set, initial_labeled_indices)
    initial_train_loader = DataLoader(initial_train_subset, batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=False)
    print(
        f"--- 开始初始训练阶段, 当前样本数: {num_initial_labeled} ({num_initial_labeled / total_videos * 100:.1f}%) ---")

    # 为初始训练创建 *独立* 的优化器和调度器
    # (使用 run_random_incremental 的逻辑重置优化器)
    initial_optimizer = create_and_load_optimizers(
        net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
        momentum=args.momentum,
        ckpt_path=args.ckpt_path,
        exp_name_toload=None,
        exp_name=args.exp_name,
        snapshot=args.snapshot,
        checkpointer=False,  # 临时计算，不加载 checkpoint
        load_opt=False  # 临时计算，不加载 optimizer 状态
    )[0]
    initial_scheduler = ExponentialLR(initial_optimizer, gamma=args.gamma)
    best_record = {'top1_acc': 0.0}

    # 训练至收敛 (使用 train_har_classifier)
    _, initial_val_acc = train_har_classifier(args, 0, initial_train_loader, net,
                                              criterion, initial_optimizer, val_loader, best_record,
                                              logger=None, scheduler=initial_scheduler, schedulerP=None,
                                              final_train=True)  # 训练至收敛

    print(f"初始训练完成！验证集准确率: {initial_val_acc:.4f}")

    # 将此初始准确率设置为 RL 奖励计算的基线
    past_val_acc = initial_val_acc

    # 记录所有已达到的初始里程碑
    next_milestone_idx = 0
    while next_milestone_idx < len(milestone_counts) and num_initial_labeled >= milestone_counts[next_milestone_idx]:
        ratio_percent = milestone_ratios[next_milestone_idx] * 100
        print(f"初始数据已达到或超过里程碑: {ratio_percent:.0f}%")
        accuracy_records.append({'ratio': ratio_percent, 'count': num_initial_labeled, 'accuracy': initial_val_acc})
        milestone_logger.append([ratio_percent, num_initial_labeled, initial_val_acc])
        next_milestone_idx += 1

    # --- 6. 增量主动学习循环 (5% -> 35%) ---
    if train_set.get_num_labeled_videos() >= max_budget:
        num_al_steps = 0
    else:
        # 计算需要多少轮 AL 才能达到 35%
        num_al_steps = math.ceil((max_budget - train_set.get_num_labeled_videos()) / args.num_each_iter)

    print(f"\n--- 开始 RL (△ACC Reward) 增量学习循环，共 {num_al_steps} 轮，直到达到 35% ({max_budget} 个样本) ---")

    for i in range(num_al_steps):
        num_labeled_before = train_set.get_num_labeled_videos()
        print(f'\n----- RL (△ACC) 选择第 {i + 1}/{num_al_steps} 轮: 当前已标注 {num_labeled_before}/{max_budget} -----')

        # 检查是否已达到最终预算
        if num_labeled_before >= max_budget:
            print(f"已达到 {max_budget} (35%) 预算，停止选择。")
            break

        # a. 获取当前状态 (来自 run_acc_reward.py)
        current_state, candidate_indices, _ = compute_state_for_har(
            args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
        )

        # b. RL智能体选择动作 (来自 run_acc_reward.py)
        # 确保选择的数量不超过预算
        num_remaining_budget = max_budget - num_labeled_before
        num_to_select_this_iter = min(args.num_each_iter, num_remaining_budget)

        if num_to_select_this_iter <= 0:
            print("预算不足以进行下一次迭代，停止。")
            break

        # 确保动作数量不超过候选集大小
        num_candidates = len(candidate_indices)
        if num_candidates == 0:
            print("没有候补视频了，停止。")
            break
        if num_candidates < args.num_each_iter:
            print(f"警告: 候补集数量 ({num_candidates}) 小于 {args.num_each_iter}，仅选择 {num_candidates} 个。")

        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)

        # 确保选择的数量不超剩余预算
        if len(action) > num_to_select_this_iter:
            action = action[:num_to_select_this_iter]  # 截断

        actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

        # c. 计算真实奖励 (△ACC) (来自 run_acc_reward.py)
        print("正在计算真实奖励 (△ACC)...")
        net_copy_for_reward = deepcopy(net)
        # 为副本创建独立、短时微调的优化器
        optimizer_for_reward = torch.optim.SGD(net_copy_for_reward.parameters(), lr=args.lr)  # 或者使用 args.lr_finetune
        temp_set_for_reward = deepcopy(train_set)

        add_labeled_videos(args, [], actual_video_ids_to_label, temp_set_for_reward, budget=max_budget, n_ep=i)
        temp_loader_for_reward = DataLoader(
            Subset(temp_set_for_reward, list(temp_set_for_reward.labeled_video_ids)),
            batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

        # 使用 *train_har_for_reward* (短时微调) 来计算新准确率
        _, new_val_acc = train_har_for_reward(net_copy_for_reward, temp_loader_for_reward, val_loader,
                                              optimizer_for_reward, criterion, args)

        real_reward = new_val_acc - past_val_acc
        print(f"真实奖励 (△ACC): {real_reward:.4f} (new: {new_val_acc:.4f} vs past: {past_val_acc:.4f})")
        del net_copy_for_reward, optimizer_for_reward, temp_set_for_reward, temp_loader_for_reward
        torch.cuda.empty_cache()

        # d. 将选中的视频加入 *实际的* 已标注集合 (来自 run_acc_reward.py)
        add_labeled_videos(args, [], actual_video_ids_to_label, train_set,
                           budget=max_budget, n_ep=i)

        # e. 计算下一个状态 (来自 run_acc_reward.py)
        next_state = None
        if train_set.get_num_labeled_videos() < max_budget and len(train_set.get_candidates_video_ids()) > 0:
            next_state, _, _ = compute_state_for_har(
                args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
            )

        # f. 将经验存入Replay Buffer (来自 run_acc_reward.py)
        reward_tensor = torch.tensor([real_reward], dtype=torch.float, device='cuda')
        memory.push(current_state['pool'], current_state['subset'], action,
                    next_state['pool'] if next_state is not None else None,
                    next_state['subset'] if next_state is not None else None,
                    reward_tensor)
        # g. 优化策略网络 (来自 run_acc_reward.py)
        if len(memory) >= args.dqn_bs:
            print("正在优化策略网络 (Policy Net)...")
            optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                BATCH_SIZE=args.dqn_bs)

        # h. 更新目标网络 (来自 run_acc_reward.py)
        if i % TARGET_UPDATE == 0:
            print('更新目标网络 (Target Net)...')
            target_net.load_state_dict(policy_net.state_dict())

        # --- i. 核心修改: 训练 HAR 分类器并记录里程碑 ---
        # (来自 run_random_incremental.py)
        print('在扩充后的数据集上训练 HAR 网络以评估里程碑...')
        current_labeled_indices = list(train_set.labeled_video_ids)
        train_subset = Subset(train_set, current_labeled_indices)
        current_train_loader = DataLoader(train_subset, batch_size=args.train_batch_size, shuffle=True,
                                          num_workers=args.workers, drop_last=False)

        # 为了准确评估该数据量下的表现，重置优化器和调度器
        print("为当前里程碑评估重置优化器和学习率调度器...")
        optimizer_milestone = create_and_load_optimizers(
            net=net, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
            momentum=args.momentum,
            ckpt_path=args.ckpt_path,
            exp_name_toload=None,  # 不加载
            exp_name=args.exp_name,
            snapshot=args.snapshot,  # 传递 snapshot 参数
            checkpointer=False,  # 不保存
            load_opt=False  # 不加载
        )[0]
        scheduler_milestone = ExponentialLR(optimizer_milestone, gamma=args.gamma)
        best_record_milestone = {'top1_acc': 0.0}  # 重置最佳记录

        # 使用 *train_har_classifier* (训练至收敛)
        # 注意：这会更新主 'net' 的权重
        _, val_acc_after_train = train_har_classifier(args, 0, current_train_loader, net,
                                                      criterion, optimizer_milestone, val_loader,
                                                      best_record_milestone,
                                                      logger=None, scheduler=scheduler_milestone, schedulerP=None,
                                                      final_train=True)  # 训练至收敛

        num_labeled_after = train_set.get_num_labeled_videos()
        print(f"训练/微调后，已标注 {num_labeled_after} 个样本, 验证集准确率为: {val_acc_after_train:.4f}")

        # j. 更新 RL 奖励的基线
        # (我们使用刚刚训练到收敛的准确率作为下一次计算奖励的基准)
        past_val_acc = val_acc_after_train

        # k. 检查并记录里程碑
        # (来自 run_random_incremental.py)
        while next_milestone_idx < len(milestone_counts) and num_labeled_after >= milestone_counts[next_milestone_idx]:
            ratio_percent = milestone_ratios[next_milestone_idx] * 100
            print(f"--- 达到新的里程碑: {ratio_percent:.0f}% ---")

            accuracy_records.append(
                {'ratio': ratio_percent, 'count': num_labeled_after, 'accuracy': val_acc_after_train})
            milestone_logger.append([ratio_percent, num_labeled_after, val_acc_after_train])

            next_milestone_idx += 1

        # 检查是否已完成所有里程碑
        if next_milestone_idx >= len(milestone_counts):
            print(f"已完成所有里程碑（达到 {milestone_ratios[-1] * 100:.0f}%）。")
            break

    # --- 7. 结束并打印最终结果 ---
    # (来自 run_random_incremental.py)
    milestone_logger.close()
    print("\n" + "=" * 50)
    print(f"       RL (△ACC Reward) 增量评估 (5% - {milestone_ratios[-1] * 100:.0f}%) 完成！")
    print("=" * 50)
    print(f"{'Data Ratio (%)':<15}{'Sample Count':<15}{'Validation Accuracy':<20}")
    print("-" * 50)
    for record in accuracy_records:
        print(f"{record['ratio']:<14.0f}%{record['count']:<15}{record['accuracy']:.4f}")
    print("-" * 50)
    print(f"详细日志已保存至: {os.path.join(args.ckpt_path, args.exp_name, log_file_name)}")

    # 保存最终的策略网络 (来自 run_acc_reward.py)
    final_policy_path = os.path.join(args.ckpt_path, args.exp_name, 'policy_net_final_incremental.pth')
    torch.save(policy_net.state_dict(), final_policy_path)
    print(f"最终策略网络已保存至: {final_policy_path}")


if __name__ == '__main__':
    args = parser.get_arguments()
    # 确保您的配置文件 (e.g., --config your_acc_reward_config.yaml)
    # 包含了 run_acc_reward.py 所需的所有 RL 相关参数
    # (如 dqn_bs, dqn_gamma, rl_buffer, lr_dqn 等)
    # 并且也包含 run_random_incremental.py 所需的参数 (如 num_each_iter)
    main(args)