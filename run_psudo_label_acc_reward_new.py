# -*- coding: utf-8 -*-
# wcf00317/alrl/alrl-reward_model/run_rl_with_pseudolabel_reward.py
#
# 严格贴近 MGRAL：用“无标签的性能代理”做奖励，只是将 mAP -> ACC。
# 奖励 = ΔACC_proxy：
#   - 在未标注 proxy-eval 子集上，用当前模型的输出计算 ACC 的无标签近似（top1置信度期望，做了类数归一化）。
#   - 将“已标注 + 本轮选择(用当前模型自身预测的伪标签)”用于一次自训练微调后，重新在同一子集上计算 ACC_proxy。
#   - 奖励 = 新的 ACC_proxy - 旧的 ACC_proxy。
# 奖励计算全程：不使用任何真实标签；不使用 KL；模型只有一个（微调时的 deepcopy 不引入新结构）。
#
# 其余训练/日志逻辑与原版一致。

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
from torch.utils.data import Subset, DataLoader, ConcatDataset

# --- 复用项目模块 ---
from models.model_utils import create_models, compute_state_for_har, select_action_for_har, \
    add_labeled_videos, optimize_model_conv
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_buffer import ReplayMemory
import utils.parser as parser
# run_rl_with_alrm 中的 train_har_classifier 是需要的，但 train_har_for_reward 我们将重写
from run_rl_with_alrm import train_har_classifier

cudnn.benchmark = False
cudnn.deterministic = True


# ==========================================================
# ✅ BUG修复: 重写的 train_har_for_reward 函数
#    这个版本可以安全地处理 val_loader 为 None 的情况。
# ==========================================================
def train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args):
    """
    一个简化的训练函数，仅运行几个epoch以获取用于计算奖励的验证分数。
    这个函数是专门为主动学习的奖励计算而设计的，追求速度而非模型的完全收敛。
    增加了对 val_loader 为 None 的处理。
    """
    # ==================== 训练部分 ====================
    net.train()
    for epoch in range(args.al_train_epochs):
        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]

            optimizer.zero_grad()

            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)
            labels_repeated = labels.repeat_interleave(num_clips)
            loss = criterion(outputs, labels_repeated)
            loss.backward()
            optimizer.step()

    # 如果没有提供 val_loader，则跳过验证阶段
    if val_loader is None:
        return 0.0, 0.0

    # ==================== 验证部分 ====================
    net.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            batch_size = inputs.shape[0]
            num_clips = inputs.shape[1]
            outputs = net(inputs, return_loss=False)
            outputs = net.cls_head(outputs)

            outputs_reshaped = outputs.view(batch_size, num_clips, -1)
            final_outputs = outputs_reshaped.mean(dim=1)

            preds = final_outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += batch_size

    if val_total == 0:
        vl_acc = 0.0
    else:
        vl_acc = val_correct / val_total

    return 0.0, vl_acc


# ============ 工具：采样未标注的 proxy-eval 子集 ============
def _sample_proxy_eval_indices(train_set, selected_batch_indices, max_size=512, rng=None):
    if rng is None:
        rng = random
    labeled = set(list(train_set.labeled_video_ids))
    selected = set(selected_batch_indices)
    pool_ids = list(train_set.get_candidates_video_ids())
    proxy_pool = [i for i in pool_ids if (i not in labeled and i not in selected)]
    if len(proxy_pool) == 0:
        return []
    rng.shuffle(proxy_pool)
    return proxy_pool[:min(max_size, len(proxy_pool))]


# ============ 工具：批量前向，返回 softmax 概率 ============
def _batched_model_probs(model, dataset, indices, device, bs=32):
    probs_all = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            part = indices[i:i + bs]
            if not part:
                continue
            clips = []
            for vid_idx in part:
                clip = dataset.get_video(vid_idx).to(device).unsqueeze(0)  # [1,C,T,H,W]
                clips.append(clip)
            clips = torch.cat(clips, dim=0)  # [B,C,T,H,W]
            logits = model(clips, return_loss=False)
            logits = model.cls_head(logits)
            probs = torch.softmax(logits, dim=1)
            probs_all.append(probs)
    if len(probs_all) == 0:
        return torch.empty(0, 0, device=device)
    return torch.cat(probs_all, dim=0)


# ============ 无标签 ACC 代理：top1 置信度的期望（类数归一） ============
def _proxy_acc_from_probs(probs, num_classes):
    """
    无标签 ACC 估计（不看 GT）：
      proxy_acc = E_x [ (max_prob(x) - 1/C) / (1 - 1/C) ]
    含义：若完全随机，max_prob 的期望 ~ 1/C；若完美，接近 1。
    """
    if probs.numel() == 0:
        return 0.0
    with torch.no_grad():
        maxp = torch.max(probs, dim=1).values  # [N]
        # 线性映射到 [0,1]
        c = float(num_classes)
        proxy = (maxp - 1.0 / c) / (1.0 - 1.0 / c)
        proxy = torch.clamp(proxy, 0.0, 1.0)
        return float(proxy.mean().item())


# ============ 生成选中样本的伪标签（来自当前模型本身） ============
def _pseudo_labels_for_indices(model, dataset, indices, device, bs=32):
    labels = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(indices), bs):
            part = indices[i:i + bs]
            if not part:
                continue
            clips = []
            for vid_idx in part:
                clip = dataset.get_video(vid_idx).to(device).unsqueeze(0)
                clips.append(clip)
            clips = torch.cat(clips, dim=0)
            logits = model(clips, return_loss=False)
            logits = model.cls_head(logits)
            pseudo = torch.argmax(logits, dim=1)  # 不做softmax也可
            labels.append(pseudo)
    if not labels:
        return []
    return torch.cat(labels, dim=0).tolist()


# ==========================================================
# 核心奖励：ΔACC_proxy（无标签、无 KL、单模型）
# ==========================================================
def calculate_pseudo_label_reward(net, selected_batch_indices, train_set, _val_loader_unused, _criterion_unused,
                                  _base_acc_unused, args):
    """
    奖励计算（严格贴合 MGRAL 思路）：
      1) 在未标注池采样 proxy-eval 子集 S（不含已标注与本轮选择）
      2) 计算当前模型在 S 上的 ACC_proxy(base) —— 无标签：取 top1 置信度期望并做类数归一
      3) 将“已标注 + 本轮选择(用当前模型自身预测的伪标签)”合并，微调当前模型的拷贝若干步
      4) 计算该拷贝在 S 上的 ACC_proxy(new)
      5) 奖励 = new - base
    全流程不使用真实标签；不使用 KL；只有一个模型（训练用 deepcopy 不算新增结构）。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = args.num_classes
    proxy_eval_size = getattr(args, 'proxy_eval_size', 512)
    proxy_eval_bs = getattr(args, 'proxy_eval_bs', 32)

    print("正在通过无标签 ACC 代理估算奖励 (ΔACC_proxy, 无泄漏, 无 KL)...")

    # 1) proxy-eval 子集
    proxy_eval_indices = _sample_proxy_eval_indices(train_set, selected_batch_indices, max_size=proxy_eval_size)
    if len(proxy_eval_indices) == 0:
        print("  - 警告：未找到可用的 proxy-eval 子集，奖励置 0")
        return 0.0, 0.0

    # 2) base ACC_proxy（当前模型在 S 上）
    base_probs = _batched_model_probs(net, train_set, proxy_eval_indices, device, bs=proxy_eval_bs)
    base_proxy_acc = _proxy_acc_from_probs(base_probs, num_classes)

    # 3) 构造微调数据：已标注 + 本轮选择(伪标签=当前模型对这些样本的预测)
    net_copy = deepcopy(net).to(device)
    optimizer_temp = torch.optim.SGD(net_copy.parameters(), lr=args.lr)

    labeled_subset = Subset(train_set, list(train_set.labeled_video_ids))
    pseudo_labeled_subset = Subset(train_set, selected_batch_indices)

    # 深拷贝后覆写标签为当前模型自身预测（自训练）
    pseudo_labeled_subset.dataset = deepcopy(pseudo_labeled_subset.dataset)
    sel_pseudo_labels = _pseudo_labels_for_indices(net, train_set, selected_batch_indices, device, bs=proxy_eval_bs)
    for i, original_idx in enumerate(selected_batch_indices):
        for j, info in enumerate(pseudo_labeled_subset.dataset.video_list_info):
            if info[2] == original_idx:  # (video_name, label, original_index)
                pseudo_labeled_subset.dataset.video_list_info[j] = (info[0], str(sel_pseudo_labels[i]), info[2])
                break

    combined_dataset = ConcatDataset([labeled_subset, pseudo_labeled_subset])
    temp_loader = DataLoader(combined_dataset, batch_size=args.train_batch_size, shuffle=True,
                             num_workers=args.workers)

    # 这里沿用你的微调函数；val_loader/criterion 对奖励无关（不读取真实标签）
    _ = train_har_for_reward(net_copy, temp_loader, None, torch.optim.SGD(net_copy.parameters(), lr=args.lr),
                             nn.CrossEntropyLoss().to(device), args)

    # 4) new ACC_proxy（微调后的拷贝在 S 上）
    new_probs = _batched_model_probs(net_copy, train_set, proxy_eval_indices, device, bs=proxy_eval_bs)
    new_proxy_acc = _proxy_acc_from_probs(new_probs, num_classes)

    reward = new_proxy_acc - base_proxy_acc
    print(f"  - ACC_proxy(base) = {base_proxy_acc:.6f} | ACC_proxy(new) = {new_proxy_acc:.6f} | ΔACC_proxy = {reward:.6f}")

    # 资源释放
    del net_copy, optimizer_temp, temp_loader, combined_dataset, pseudo_labeled_subset, labeled_subset
    torch.cuda.empty_cache()

    # 第二返回值用于兼容上层日志（这里返回 new_proxy_acc）
    return reward, new_proxy_acc


def main(args):
    # --- 配置加载 ---
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

    # 注意：val_loader 仅用于训练日志/早停，奖励计算不依赖它
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
        print('--- 启动 RL 训练 (奖励信号: ΔACC_proxy，无泄漏/无KL/单模型) ---')

        Transition = namedtuple('Transition', (
            'state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))
        memory = ReplayMemory(args.rl_buffer)
        TARGET_UPDATE = 5
        steps_done = 0
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

        # （仅日志用途，不参与奖励）
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
        print(f"初始验证准确率 (仅日志)：{past_val_acc:.4f}")

        num_al_steps = (args.budget_labels - train_set.get_num_labeled_videos()) // args.num_each_iter

        for i in range(num_al_steps):
            print(f'\n--------------- RL (ΔACC_proxy Reward) 回合 {i + 1}/{num_al_steps} ---------------')

            current_state, candidate_indices, _ = compute_state_for_har(
                args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
            )
            action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
            actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

            # ✅ 使用 ΔACC_proxy 奖励（不使用真实标签）
            estimated_reward, _ = calculate_pseudo_label_reward(
                net, actual_video_ids_to_label, train_set, None, None, None, args
            )
            print(f"估算奖励 (ΔACC_proxy): {estimated_reward:.4f}")

            # 将本轮选择加入已标注集合（后续进入常规训练）
            add_labeled_videos(args, [], actual_video_ids_to_label, train_set,
                               budget=args.budget_labels, n_ep=i)

            current_labeled_indices = list(train_set.labeled_video_ids)
            train_loader = DataLoader(Subset(train_set, current_labeled_indices),
                                      batch_size=args.train_batch_size, shuffle=True,
                                      num_workers=args.workers, drop_last=False)

            print('使用新选择的视频更新主HAR网络...')
            _, past_val_acc = train_har_for_reward(net, train_loader, val_loader, optimizer, criterion, args)
            print(f"主模型更新后 (日志) 准确率: {past_val_acc:.4f}")

            next_state = None
            if train_set.get_num_labeled_videos() < args.budget_labels:
                next_state, _, _ = compute_state_for_har(
                    args, net, train_set, train_set.get_candidates_video_ids(), list(train_set.labeled_video_ids)
                )

            reward_tensor = torch.tensor([estimated_reward], dtype=torch.float, device='cuda')
            memory.push(current_state, action, next_state, reward_tensor)

            if len(memory) >= args.dqn_bs:
                optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP,
                                    GAMMA=args.dqn_gamma, BATCH_SIZE=args.dqn_bs)

            if i % TARGET_UPDATE == 0:
                print('更新目标网络...')
                target_net.load_state_dict(policy_net.state_dict())

        # --- 收敛阶段（与原流程一致，仅用于最终质量评估/日志） ---
        print("\n预算已用尽。在所有已选数据上训练HAR模型至收敛...")
        logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                             log_name='final_convergence_log.txt')
        final_labeled_indices = list(train_set.labeled_video_ids)
        final_train_loader = DataLoader(Subset(train_set, final_labeled_indices), batch_size=args.train_batch_size,
                                        shuffle=True, num_workers=args.workers, drop_last=False)
        _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net,
                                                criterion, optimizer, val_loader,
                                                best_record, logger, scheduler,
                                                schedulerP, final_train=True)
        print(f"收敛后的最终验证集准确率: {final_val_acc:.4f}")
        torch.save(policy_net.state_dict(), os.path.join(args.ckpt_path, args.exp_name, 'policy_final.pth'))


if __name__ == '__main__':
    args = parser.get_arguments()
    # 可在 config/命令行加入：
    #   --proxy_eval_size 512 --proxy_eval_bs 32
    # 调整无标签评估子集规模与评估批大小
    main(args)