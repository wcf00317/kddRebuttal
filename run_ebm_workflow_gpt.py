# 文件名: run_ebm_workflow.py

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
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import pickle

# --- 核心导入 ---
from models.model_utils import create_models, add_labeled_videos, compute_state_for_har, select_action_for_har, optimize_model_conv, debug_optimize_model_conv
from utils.feature_extractor import UnifiedFeatureExtractor
from torch.utils.data import Subset, DataLoader
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.gpt_replay_buffer import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward
import utils.al_scoring as scoring

# --- 导入我们新建的EBM训练模块 ---
from models.train_ebm import train_ebm_reward_model, load_ebm_scorer, predict_ebm_reward

cudnn.benchmark = False
cudnn.deterministic = True


import sys

class VerboseLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        try:
            # 有些文件系统（如HDFS挂载）不支持写操作
            self.log_file = open(log_path, "a", encoding='utf-8')
            self.file_enabled = True
        except Exception as e:
            self.log_file = None
            self.file_enabled = False
            print(f"[Logger Warning] 无法打开日志文件 {log_path}: {e}")

    def write(self, message):
        # 永远输出到控制台
        try:
            self.terminal.write(message)
        except Exception:
            pass

        # 尝试写入日志文件（但如果不支持，就跳过）
        if self.file_enabled and self.log_file is not None:
            try:
                self.log_file.write(message)
            except OSError as e:
                if e.errno == 95:  # Operation not supported
                    print(f"[Logger Warning] 文件系统不支持写入日志 ({e}); 将仅输出到控制台。")
                else:
                    print(f"[Logger Warning] 写日志出错: {e}")
                self.file_enabled = False
                self.log_file = None
            except Exception as e:
                print(f"[Logger Warning] 写日志出错: {e}")
                self.file_enabled = False
                self.log_file = None

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        if self.file_enabled and self.log_file is not None:
            try:
                self.log_file.flush()
            except Exception:
                pass

    def __del__(self):
        try:
            if self.log_file:
                self.log_file.close()
        except Exception:
            pass

def get_available_strategies(args):
    """根据配置文件动态获取所有启用的策略及其对应的评分函数。"""
    

    strategy_map = {
        'use_statistical_features': ('entropy', scoring.compute_entropy_score),
        'use_diversity_feature': ('diversity', scoring.compute_diversity_score),
        'use_representativeness_feature': ('representativeness', scoring.compute_representativeness_score),
        'use_prediction_margin_feature': ('prediction_margin', scoring.compute_prediction_margin_score),
        'use_labeled_distance_feature': ('labeled_distance', scoring.compute_labeled_distance_score),
        'use_neighborhood_density_feature': ('neighborhood_density', scoring.compute_neighborhood_density_score),
        'use_temporal_consistency_feature': ('temporal_consistency', scoring.compute_temporal_consistency_score),
        'use_cross_view_consistency_feature': ('cross_view_consistency', scoring.compute_cross_view_consistency_score)
    }
    available_strategies = [
        ('bald', scoring.compute_bald_score),
        ('egl', scoring.compute_egl_adaptive_topk)
    ]
    for arg_name, (score_name, score_func) in strategy_map.items():
        if getattr(args, arg_name, False):
            available_strategies.append((score_name, score_func))
    print(f"启用的策略共 {len(available_strategies)} 个: {[name for name, _ in available_strategies]}")
    return available_strategies


def main():
    args = parser.get_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.exp_name = f"{args.exp_name}_{timestamp}"
    exp_dir = os.path.join(args.ckpt_path, args.exp_name)
    check_mkdir(args.ckpt_path)
    check_mkdir(exp_dir)

    sys.stdout = VerboseLogger(os.path.join(exp_dir, 'verbose_run_log.txt'))
    parser.save_arguments(args)
    print(f"实验将保存在: {exp_dir}")

    # =====================================================================
    # STAGE 1: 收集偏好数据
    # =====================================================================
    print("\n" + "=" * 25 + "  STAGE 1: PREFERENCE DATA COLLECTION  " + "=" * 25)
    feature_extractor = UnifiedFeatureExtractor(args)
    net_stage1, _, _ = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                     model_ckpt_path=args.model_ckpt_path, num_classes=args.num_classes,
                                     use_policy=False,
                                     embed_dim=args.embed_dim)
    net_stage1.cuda()
    aug_level = args.augment_level if getattr(args, 'use_cross_view_consistency_feature', False) else None

    _, train_set, _, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        augment_level=aug_level,initial_labeled_ratio=args.initial_labeled_ratio,model_type=args.model_type
    )

    unlabeled_indices = train_set.get_candidates_video_ids()
    if not unlabeled_indices:
        raise ValueError("未标注池为空，无法进行数据收集。")

    strategies = get_available_strategies(args)
    precomputed_data = scoring.precompute_data_for_scoring(args, net_stage1, unlabeled_indices, train_set,
                                                           batch_size=args.val_batch_size)

    all_scores_map = {}
    for name, fn in tqdm(strategies, desc="计算全局分数"):
        if name in ['bald', 'egl']:
            all_scores_map[name] = fn(net_stage1, unlabeled_indices, train_set)
        else:
            all_scores_map[name] = fn(precomputed_data)
    print("全局分数计算完成。")

    alrm_preference_data = []

    num_pairs_per_strategy = 5 # 假设我们希望每个策略生成 5 对偏好
    
    for name, _ in tqdm(strategies, desc="提取特征两端样本"):
        scores = all_scores_map[name]
        k = min(args.num_each_iter, len(unlabeled_indices) // num_pairs_per_strategy) # 每个批次的样本数
        
        # 确保 k 至少为 1
        if k == 0:
            print(f"警告: 策略 {name} 的批次大小 k 为 0，可能候选池太小或 num_pairs_per_strategy 过大。")
            k = 1 # 至少选一个
            
        # 增加循环，为每个策略提取 num_pairs_per_strategy 对偏好
        for i in range(num_pairs_per_strategy):
            # 这里采取一个更通用的做法：
            # 将所有样本按分数排序，然后从不同的分段中选择 winner 和 loser
            sorted_scores, sorted_indices = torch.sort(scores, descending=True) # 分数从高到低
            
            # 定义 winner 和 loser 的索引范围
            # 假设我们想从顶部 10% 和底部 10% 中选择
            top_percent = 0.05 + 0.02 * i       # 从 5% → 13%
            bottom_percent = 0.05 + 0.02 * (num_pairs_per_strategy - 1 - i)
            
            num_total = len(unlabeled_indices)
            num_top_samples = int(num_total * top_percent)
            num_bottom_samples = int(num_total * bottom_percent)
            
            # 确保有足够的样本来选择 k 个
            num_top_samples = max(k, num_top_samples)
            num_bottom_samples = max(k, num_bottom_samples)

            # 确保不会超出范围
            if num_top_samples > num_total: num_top_samples = num_total
            if num_bottom_samples > num_total: num_bottom_samples = num_total

            # Winner 候选池：从排序后的顶部 `num_top_samples` 中选择
            winner_candidate_indices = sorted_indices[:num_top_samples]
            # Loser 候选池：从排序后的底部 `num_bottom_samples` 中选择
            loser_candidate_indices = sorted_indices[num_total - num_bottom_samples:]
            
            # 确保候选池有足够的样本
            if len(winner_candidate_indices) < k or len(loser_candidate_indices) < k:
                print(f"警告: 策略 {name} 第 {i+1} 轮抽取时候选池不足，跳过此轮或选择更少样本。")
                if len(winner_candidate_indices) == 0 or len(loser_candidate_indices) == 0: continue
                # 如果不足k，则选择所有可用的
                current_k = min(len(winner_candidate_indices), len(loser_candidate_indices), k)
            else:
                current_k = k
                
            # 随机选择 k 个索引
            winner_sample_indices = random.sample(winner_candidate_indices.tolist(), current_k)
            loser_sample_indices = random.sample(loser_candidate_indices.tolist(), current_k)

            winner_idx = [unlabeled_indices[idx] for idx in winner_sample_indices]
            loser_idx = [unlabeled_indices[idx] for idx in loser_sample_indices]
            
            # 确保 winner_idx 和 loser_idx 不为空
            if not winner_idx or not loser_idx:
                continue

            winner_scores = {sname: all_scores_map[sname][winner_sample_indices] for sname, _ in strategies}
            loser_scores = {sname: all_scores_map[sname][loser_sample_indices] for sname, _ in strategies}
            
            # 确保 batch_scores 中的张量有正确的维度
            winner_feat = feature_extractor.extract(winner_idx, net_stage1, train_set, batch_scores=winner_scores)
            loser_feat = feature_extractor.extract(loser_idx, net_stage1, train_set, batch_scores=loser_scores)
            
            alrm_preference_data.append({'winner': winner_feat, 'loser': loser_feat})
            
        pairs_added = len(alrm_preference_data) - prev_len
        print(f"策略 {name} 完成，新增 {pairs_added} 对偏好样本 (累计 {len(alrm_preference_data)} 对)。")
        prev_len = len(alrm_preference_data)


    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    with open(alrm_data_path, 'wb') as f:
        pickle.dump(alrm_preference_data, f)
    print(f"偏好数据已保存至 {alrm_data_path}")
    del net_stage1, train_set, precomputed_data
    torch.cuda.empty_cache()

    # =====================================================================
    # STAGE 2: 训练 EBM 奖励模型
    # =====================================================================
    print("\n" + "=" * 25 + "  STAGE 2: EBM REWARD MODEL TRAINING  " + "=" * 25)
    if not train_ebm_reward_model(alrm_preference_data, exp_dir,
        feature_names=feature_extractor.feature_dim_names  # <--- 新增这个参数传递
    ):
        print("EBM 奖励模型训练失败，退出。")
        return
    print("EBM 奖励模型训练完成。")
    del alrm_preference_data
    torch.cuda.empty_cache()

    # =====================================================================
    # STAGE 3: RL 智能体训练
    # =====================================================================
    print("\n" + "=" * 25 + "  STAGE 3: RL AGENT TRAINING WITH EBM  " + "=" * 25)
    net_stage3, policy_net, target_net = create_models(dataset=args.dataset, model_cfg_path=args.model_cfg_path,
                                                       model_ckpt_path=args.model_ckpt_path,
                                                       num_classes=args.num_classes, use_policy=True,
                                                       embed_dim=args.embed_dim)

    _, train_set_rl, val_loader, _ = get_data(
        data_path=args.data_path, tr_bs=args.train_batch_size, vl_bs=args.val_batch_size,
        dataset_name=args.dataset, n_workers=args.workers, clip_len=args.clip_len,
        augment_level=aug_level,initial_labeled_ratio=args.initial_labeled_ratio,model_type=args.model_type
    )

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer_rl, optimizerP = create_and_load_optimizers(
        net=net_stage3, opt_choice=args.optimizer, lr=args.lr, wd=args.weight_decay,
        momentum=args.momentum, policy_net=policy_net, lr_dqn=args.lr_dqn,
        ckpt_path=args.ckpt_path, exp_name=args.exp_name,
        exp_name_toload=args.exp_name_toload,  # 从args中获取
        snapshot=args.snapshot,  # 从args中获取
        checkpointer=args.checkpointer,  # 从args中获取
        load_opt=args.load_opt,  # 从args中获取
    )

    ebm_scorer = load_ebm_scorer(exp_dir)

    # ✅ 新 Transition 定义：双流结构
    Transition = namedtuple('Transition', (
        'state_pool', 'state_subset', 'action',
        'next_state_pool', 'next_state_subset', 'reward'
    ))
    memory = ReplayMemory(args.rl_buffer)

    TARGET_UPDATE = 2
    steps_done = 0
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    scheduler = ExponentialLR(optimizer_rl, gamma=args.gamma)
    schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

    num_al_steps = (args.budget_labels - train_set_rl.get_num_labeled_videos()) // args.num_each_iter
    policy_init_flat = None

    for i in range(num_al_steps):
        print(f'\n--- RL训练回合 {i + 1}/{num_al_steps} ---')

        current_state, candidate_indices, _ = compute_state_for_har(
            args, net_stage3, train_set_rl,
            train_set_rl.get_candidates_video_ids(),
            list(train_set_rl.labeled_video_ids)
        )

        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
        actual_video_ids = [candidate_indices[idx] for idx in action.tolist()]

        batch_scores = {}
        precomputed_batch = scoring.precompute_data_for_scoring(args, net_stage3, actual_video_ids,
                                                                train_set_rl, batch_size=args.train_batch_size)
        for sname, fn in get_available_strategies(args):
            if sname not in ['bald', 'egl']:
                batch_scores[sname] = fn(precomputed_batch)

        batch_features = feature_extractor.extract(actual_video_ids, net_stage3, train_set_rl,
                                                   batch_scores=batch_scores).cuda()

        predicted_rewards = predict_ebm_reward(ebm_scorer, batch_features)  # shape [K]
    #     print(f"EBM预测奖励: mean={predicted_rewards.mean():.4f}, "
    #   f"min={predicted_rewards.min():.4f}, max={predicted_rewards.max():.4f}, "
    #   f"std={predicted_rewards.std():.4f}")

        add_labeled_videos(args, [], actual_video_ids, train_set_rl, args.budget_labels, i)

        train_loader_rl = DataLoader(Subset(train_set_rl, list(train_set_rl.labeled_video_ids)),
                                     batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, _ = train_har_for_reward(net_stage3, train_loader_rl, val_loader, optimizer_rl, criterion, args)

        next_state = None
        if train_set_rl.get_num_labeled_videos() < args.budget_labels:
            next_state, _, _ = compute_state_for_har(args, net_stage3, train_set_rl,
                                                     train_set_rl.get_candidates_video_ids(),
                                                     list(train_set_rl.labeled_video_ids))

        # ✅ Push 双流结构
        # memory.push(
        #     current_state['pool'], current_state['subset'], action,
        #     next_state['pool'] if next_state is not None else None,
        #     next_state['subset'] if next_state is not None else None,
        #     torch.tensor([predicted_reward], dtype=torch.float, device='cuda')
        # )
        # ✅ 每个选中样本都单独 push 一条 transition
        # 先构建从全局视频ID到当前池行号的映射
        vid_to_row = {vid: r for r, vid in enumerate(candidate_indices)}
        assert len(actual_video_ids) == len(predicted_rewards)
        for vid, reward_val in zip(actual_video_ids, predicted_rewards):
        # for vid in actual_video_ids:
            row = vid_to_row.get(int(vid), None)
            if row is None:
                print(f"[ERROR] vid {vid} 不在 candidate_indices 中，跳过该条样本。")
                continue

            # 从池中取该动作的特征向量（[4096]），并扩一维成 [1,4096]
            action_embed = current_state['pool'][row].unsqueeze(0).cuda(non_blocking=True)  # [1, 4096]

            # next_state_pool 要求是“下一个状态下的**候选动作池**”，形如 [K,4096]
            # 你这里已经传 next_state['pool'] 了，若它现在是 [N_pool',4096] 就正合适
            memory.push(
                action_embed,                                          # 当前动作特征
                current_state['subset'].unsqueeze(0).cuda(),           # 当前状态
                torch.tensor([vid], device='cuda'),                    # 动作 ID
                next_state['pool'].unsqueeze(0) if next_state else None,   # ✅ 增加 batch 维度 → [1, K, 4096]
                next_state['subset'].unsqueeze(0) if next_state else None, # ✅ 统一为 [1, M, 4096]
                reward_val.unsqueeze(0).to('cuda')
                # torch.tensor([predicted_reward], dtype=torch.float, device='cuda')
            )
        print(f"len(memory): {len(memory)}, BATCH_SIZE: {args.dqn_bs}")
        if len(memory) >= args.dqn_bs:
            for _ in range(args.dqn_opt_per_iter):
                optimize_model_conv(
                    args, memory, Transition, policy_net, target_net, optimizerP,
                    GAMMA=args.dqn_gamma, BATCH_SIZE=args.dqn_bs
                )

                if i % 10 == 0:
                    print(f"[INFO] Step {i} | loss={getattr(optimize_model_conv, 'last_loss', None)}")
                    print(f"[INFO] Policy Q mean={getattr(optimize_model_conv, 'last_q_mean', None)} | Reward mean={getattr(optimize_model_conv, 'last_reward_mean', None)}")

        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 🧠 打印策略参数漂移以监控是否真的学习
        with torch.no_grad():
            flat_params = torch.cat([p.view(-1) for p in policy_net.parameters()])
            if policy_init_flat is None:
                policy_init_flat = flat_params.clone()
            drift = (flat_params - policy_init_flat).norm().item()
        print(f"[Policy Drift] L2 距离: {drift:.6f}")

    print("\n预算用尽，在所有已选数据上训练至收敛...")
    logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                         log_name='final_convergence_log.txt')
    final_loader = DataLoader(Subset(train_set_rl, list(train_set_rl.labeled_video_ids)),
                              batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
    _, final_val_acc = train_har_classifier(args, 0, final_loader, net_stage3, criterion,
                                            optimizer_rl, val_loader, best_record, logger,
                                            scheduler, schedulerP, final_train=True)
    logger.close()

    print(f"\n--- STAGE 3 COMPLETE --- 最终验证集准确率: {final_val_acc:.4f}")
    torch.save(policy_net.state_dict(), os.path.join(exp_dir, 'policy_final.pth'))


if __name__ == '__main__':
    main()
