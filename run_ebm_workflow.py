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
import torch.optim as optim
from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import pickle

# --- 核心导入 ---
from models.model_utils import create_models, add_labeled_videos, compute_state_for_har, select_action_for_har, \
    optimize_model_conv
from utils.feature_extractor import UnifiedFeatureExtractor, get_all_unlabeled_embeddings, get_all_labeled_embeddings
from torch.utils.data import Subset, DataLoader
from data.data_utils import get_data
from utils.final_utils import check_mkdir, create_and_load_optimizers, get_logfile
from utils.replay_                                                                                                                                                                                             import ReplayMemory
import utils.parser as parser
from run_rl_with_alrm import train_har_classifier, train_har_for_reward
import utils.al_scoring as scoring

# --- 导入我们新建的EBM训练模块 ---
from models.train_ebm import train_ebm_reward_model, load_ebm_scorer, predict_ebm_reward

cudnn.benchmark = False
cudnn.deterministic = True


# (VerboseLogger 类与 run_minimalist_tournament.py 中保持一致)
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
    # print("!!! [ABLATION MODE] Running MODEL-DRIVEN strategies ONLY !!!")
    # model_driven_strategies = [
    #     ('entropy', scoring.compute_entropy_score),
    #     ('prediction_margin', scoring.compute_prediction_margin_score),
    #     ('bald', scoring.compute_bald_score),
    #     ('egl', scoring.compute_egl_adaptive_topk)
    # ]
    # print(f"启用的策略共 {len(model_driven_strategies)} 个: {[name for name, _ in model_driven_strategies]}")
    # return model_driven_strategies

    strategy_map = {
        'use_statistical_features': ('entropy', scoring.compute_entropy_score),
        'use_diversity_feature': ('diversity', scoring.compute_diversity_score),
        'use_representativeness_feature': ('representativeness', scoring.compute_representativeness_score),
        'use_prediction_margin_feature': ('prediction_margin', scoring.compute_prediction_margin_score),
        'use_labeled_distance_feature': ('labeled_distance', scoring.compute_labeled_distance_score),
        'use_neighborhood_density_feature': ('neighborhood_density', scoring.compute_neighborhood_density_score),
        'use_temporal_consistency_feature': ('temporal_consistency', scoring.compute_temporal_consistency_score),
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
    # --- 0. 初始化和配置加载 (与 run_minimalist_tournament.py 一致) ---
    args = parser.get_arguments()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.exp_name_toload:
        # 如果指定了要加载的实验，则 exp_name 就是那个完整目录名
        # 并且我们不覆盖 args.exp_name_toload
        print(f"--- 正在加载已存在的实验: {args.exp_name_toload} ---")
        # 将 args.exp_name 也设置为加载的名称，以便日志和优化器加载统一
        args.exp_name = args.exp_name_toload
        exp_dir = os.path.join(args.ckpt_path, args.exp_name_toload)
        # 确保基础路径存在，但不创建 exp_dir (因为它应该已存在)
        check_mkdir(args.ckpt_path)
        if not os.path.exists(exp_dir):
            print(f"严重警告: 尝试加载 {exp_dir}，但目录不存在！")
            # 仍然创建它，以防万一，但跳过逻辑会失败
            check_mkdir(exp_dir)
    else:
        # 如果是新实验，才添加时间戳
        print("--- 正在创建新实验 ---")
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f"{args.exp_name}_{timestamp}"
        exp_dir = os.path.join(args.ckpt_path, args.exp_name)
        check_mkdir(args.ckpt_path)
        check_mkdir(exp_dir)

    sys.stdout = VerboseLogger(os.path.join(exp_dir, 'verbose_run_log.txt'))
    parser.save_arguments(args)
    if hasattr(args, 'config') and args.config and os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(exp_dir, os.path.basename(args.config)))
    print(f"实验将保存在: {exp_dir}")

    # ===================================================================================
    #              第一阶段: 使用特征两端提取法收集偏好数据
    # ===================================================================================
    alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
    if os.path.exists(alrm_data_path):
        print(f"\n--- STAGE 1 SKIPPED --- 正在从 {alrm_data_path} 加载已有的偏好数据...")
        with open(alrm_data_path, 'rb') as f:
            alrm_preference_data = pickle.load(f)
        feature_extractor = UnifiedFeatureExtractor(args)
        aug_level = args.augment_level if getattr(args, 'use_cross_view_consistency_feature',
                                                  False) else None  # Stage 3 需要

    else:
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

        # precomputed_data = scoring.precompute_data_for_scoring(
        #     args, net_stage1, unlabeled_indices, train_set, batch_size=args.val_batch_size
        # )

        print("正在为所有策略预计算全局分数...")
        strategies_to_run = get_available_strategies(args)
        precomputed_data = scoring.precompute_data_for_scoring(
            args, net_stage1, unlabeled_indices, train_set, batch_size=args.val_batch_size
        )

        all_scores_map = {}
        for strategy_name, scoring_function in tqdm(strategies_to_run, desc="全局分数计算"):
            if strategy_name in ['bald', 'egl']:
                all_scores_map[strategy_name] = scoring_function(net_stage1, unlabeled_indices, train_set)
            else:
                all_scores_map[strategy_name] = scoring_function(precomputed_data )
        print("全局分数计算完成。")

        # --- 步骤 2: 保持您原有的For循环结构，用于筛选和特征化 ---
        print("\n开始遍历策略以提取两端批次...")
        alrm_preference_data = []

        # --- 外层循环：遍历每种策略 ---
        for strategy_name, _ in tqdm(strategies_to_run, desc="为不同策略提取特征两端样本"):
            print(f"\n--- 处理策略: {strategy_name} ---")

            # 获取当前策略对所有【原始】未标注样本的分数
            # 注意：这里的 scores 是针对【全部】未标注样本计算的，我们后续需要从中筛选
            scores = all_scores_map[strategy_name]

            if scores.numel() == 0:
                print(f"警告: 策略 {strategy_name} 未能计算出任何分数，跳过。")
                continue

            # 初始化当前策略下【可用】的样本索引列表（初始时为所有样本）
            # 我们用位置索引 (0 到 N-1) 来操作
            available_pos_indices = list(range(len(unlabeled_indices)))

            # --- 内层循环：为当前策略生成 N 对偏好数据 ---
            num_pairs_generated_for_strategy = 0
            for pair_idx in range(args.num_pairs_per_strategy):

                # 检查是否还有足够的样本可选 (至少需要 2*k 个)
                if len(available_pos_indices) < 2 * args.num_each_iter:
                    print(
                        f"  [WARN] 第 {pair_idx + 1} 对: 可用样本不足 ({len(available_pos_indices)} < {2 * args.num_each_iter})，停止为策略 {strategy_name} 生成更多对。")
                    break

                print(f"  --- 生成第 {pair_idx + 1}/{args.num_pairs_per_strategy} 对 ---")

                # 从【可用】样本的分数中选出 Top-k 和 Bottom-k
                # 1. 获取可用样本对应的分数
                current_available_scores = scores[available_pos_indices]

                # 2. 在可用分数中找 Top-k 和 Bottom-k 的【相对索引】
                k = min(args.num_each_iter, len(available_pos_indices) // 2)  # 确保 k 不超过可用的一半

                top_scores_relative, top_relative_indices = torch.topk(current_available_scores, k=k, largest=True)
                bottom_scores_relative, bottom_relative_indices = torch.topk(current_available_scores, k=k,
                                                                             largest=False)

                # 3. 将【相对索引】映射回【原始位置索引】
                top_pos_indices_original = torch.tensor([available_pos_indices[i] for i in top_relative_indices])
                bottom_pos_indices_original = torch.tensor([available_pos_indices[i] for i in bottom_relative_indices])

                # 4. 根据【原始位置索引】获取视频 ID
                winner_batch_indices = [unlabeled_indices[i] for i in top_pos_indices_original]
                loser_batch_indices = [unlabeled_indices[i] for i in bottom_pos_indices_original]

                # 5. 打包所有策略的分数（使用原始位置索引）
                winner_batch_scores = {s_name: all_scores_map[s_name][top_pos_indices_original] for s_name, _ in
                                       strategies_to_run}
                loser_batch_scores = {s_name: all_scores_map[s_name][bottom_pos_indices_original] for s_name, _ in
                                      strategies_to_run}

                # 6. 提取特征
                winner_features = feature_extractor.extract(
                    winner_batch_indices, net_stage1, train_set, batch_scores=winner_batch_scores
                )
                loser_features = feature_extractor.extract(
                    loser_batch_indices, net_stage1, train_set, batch_scores=loser_batch_scores
                )

                # 7. 添加到结果列表
                alrm_preference_data.append({'winner': winner_features, 'loser': loser_features})
                num_pairs_generated_for_strategy += 1

                # 8. 更新可用样本索引列表：移除刚刚被选中的 Winner 和 Loser
                # 注意：需要从后往前删除，或者转换成集合操作，以避免索引错位
                indices_to_remove = set(top_pos_indices_original.tolist() + bottom_pos_indices_original.tolist())
                available_pos_indices = [idx for idx in available_pos_indices if idx not in indices_to_remove]

                # --- 日志输出 (使用相对分数) ---
                print(f"    [好学生 - Winner Batch]")
                print(f"      - 样本索引 (原始): {winner_batch_indices}")
                print(f"      - 对应分数: {[f'{s:.4f}' for s in top_scores_relative.tolist()]}")  # 用相对分数
                print(f"    [差学生 - Loser Batch]")  # 改回 Loser
                print(f"      - 样本索引 (原始): {loser_batch_indices}")
                print(f"      - 对应分数: {[f'{s:.4f}' for s in bottom_scores_relative.tolist()]}")  # 用相对分数
                print(f"    剩余可用样本数: {len(available_pos_indices)}")

            print(f"--- 策略 {strategy_name} 完成，共生成 {num_pairs_generated_for_strategy} 对偏好数据 ---")
            print(f"{'=' * 50}\n")
        alrm_data_path = os.path.join(exp_dir, 'alrm_preference_data.pkl')
        with open(alrm_data_path, 'wb') as f:
            pickle.dump(alrm_preference_data, f)
        print(f"\n--- STAGE 1 COMPLETE --- 偏好数据已保存至 {alrm_data_path}，共 {len(alrm_preference_data)} 对。")
        del net_stage1, train_set, precomputed_data
        torch.cuda.empty_cache()

    # ===================================================================================
    #                         第二阶段: 训练 EBM 奖励模型
    # ===================================================================================
    scorer_path = os.path.join(exp_dir, 'ebm_scorer.pkl')
    if os.path.exists(scorer_path):
        print(f"\n--- STAGE 2 SKIPPED --- 已找到 {scorer_path}，跳过EBM训练。")
    else:
        print("\n" + "=" * 25 + "  STAGE 2: EBM REWARD MODEL TRAINING  " + "=" * 25)

        training_successful = train_ebm_reward_model(alrm_preference_data, exp_dir,
            feature_names=feature_extractor.feature_dim_names)

        if not training_successful:
            print("EBM奖励模型训练失败，工作流程终止。")
            return

        print(f"--- STAGE 2 COMPLETE ---")
        del alrm_preference_data
        torch.cuda.empty_cache()

    # ===================================================================================
    #                  第三阶段: 使用 EBM 奖励模型训练 RL 智能体
    # ===================================================================================
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

    # --- 核心修改: 加载EBM计分器 ---
    ebm_scorer = load_ebm_scorer(exp_dir)

    Transition = namedtuple('Transition',
                            ('state_pool', 'state_subset', 'action', 'next_state_pool', 'next_state_subset', 'reward'))    
    memory = ReplayMemory(args.rl_buffer)
    steps_done = 0
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    scheduler = ExponentialLR(optimizer_rl, gamma=args.gamma)
    schedulerP = ExponentialLR(optimizerP, gamma=args.gamma_scheduler_dqn)

    num_al_steps = (args.budget_labels - train_set_rl.get_num_labeled_videos()) // args.num_each_iter
    for i in range(num_al_steps):
        print(f'\n--- RL训练回合 {i + 1}/{num_al_steps} ---')
        # # --- 新增RL阶段的预计算 ---
        # rl_unlabeled_indices = train_set_rl.get_candidates_video_ids()
        #
        # all_unlabeled_embeds = get_all_unlabeled_embeddings(args, net_stage3, train_set_rl)
        # all_labeled_embeds = get_all_labeled_embeddings(args, net_stage3, train_set_rl)

        current_state, candidate_indices, _ = compute_state_for_har(args, net_stage3, train_set_rl,
                                                                    train_set_rl.get_candidates_video_ids(),
                                                                    list(train_set_rl.labeled_video_ids))
        action, steps_done, _ = select_action_for_har(args, policy_net, current_state, steps_done)
        actual_video_ids_to_label = [candidate_indices[idx] for idx in action.tolist()]

        # 3. **按需、仅为选中的批次计算所有必需的分数**
        batch_scores = {}

        # a. 预计算当前批次的所有数据
        #    注意：这里的 video_indices 参数只包含当前被选中的批次，非常高效
        precomputed_batch_data = scoring.precompute_data_for_scoring(
            args, net_stage3, actual_video_ids_to_label, train_set_rl, batch_size=args.train_batch_size
        )

        # b. 调用所有需要的评分函数
        #    get_available_strategies 帮助我们动态获取所有启用的评分函数
        strategies_to_run = get_available_strategies(args)
        for strategy_name, scoring_function in strategies_to_run:
            if strategy_name not in ['bald', 'egl']:  # BALD/EGL太慢，在RL循环中跳过
                batch_scores[strategy_name] = scoring_function(precomputed_batch_data)

        # 4. 调用 extract 函数，传入为该批次计算好的分数
        batch_features = feature_extractor.extract(
            actual_video_ids_to_label, net_stage3, train_set_rl, batch_scores=batch_scores
        ).cuda()

        # --- 核心修改: 使用 EBM 预测奖励 ---
        predicted_reward = predict_ebm_reward(ebm_scorer, batch_features)
        print(f"EBM 预测奖励 (P(x > μ)): {predicted_reward}")

        add_labeled_videos(args, [], actual_video_ids_to_label, train_set_rl, budget=args.budget_labels, n_ep=i)

        current_labeled_indices = list(train_set_rl.labeled_video_ids)
        train_loader_rl = DataLoader(Subset(train_set_rl, current_labeled_indices),
                                     batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)
        _, _ = train_har_for_reward(net_stage3, train_loader_rl, val_loader, optimizer_rl, criterion, args)

        next_state = None
        if train_set_rl.get_num_labeled_videos() < args.budget_labels:
            next_state, _, _ = compute_state_for_har(args, net_stage3, train_set_rl,
                                                     train_set_rl.get_candidates_video_ids(),
                                                     list(train_set_rl.labeled_video_ids))

        sel_idxs = action.tolist()  # 选中的样本在候选池中的行号
        pool = current_state['pool']  # [N, D]
        subset = current_state['subset']  # [M, D]

        # 下一个状态（若还有预算）
        if next_state is not None:
            next_pool = next_state['pool']  # [N_next, D]
            next_subset = next_state['subset']  # [M, D]
        else:
            next_pool, next_subset = None, None

        # 批次级 reward（标量）
        reward_tensor = predicted_reward.cuda()

        for idx in sel_idxs:
            action_embed = pool[idx]  # [D] —— “被执行的动作的向量”，样本级
            memory.push(
                action_embed,  # ← state_pool 实际承载“动作向量 [D]”
                subset.unsqueeze(0).cuda(non_blocking=True),  # [1, M, D]
                torch.tensor(idx, device='cuda'),  # 0-D 动作索引（未在优化器中使用，但保留无害）
                None if next_pool is None else next_pool.unsqueeze(0),  # [1, N_next, D] or None
                None if next_subset is None else next_subset.unsqueeze(0),  # [1, M, D]     or None
                reward_tensor.clone()  # 0-D 张量；每条 transition 同一个批次级 reward
            )

        if len(memory) >= args.dqn_bs:
            loss_val = optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, GAMMA=args.dqn_gamma,
                                BATCH_SIZE=args.dqn_bs,TAU=args.dqn_tau)
            if loss_val is not None:
                # (您现有的 debug 日志)
                print(
                    f"--- [LOG] loss={loss_val:.6f}, q_mean={getattr(optimize_model_conv, 'last_q_mean', 'N/A')}, reward_mean={getattr(optimize_model_conv, 'last_reward_mean', 'N/A')}")

    print("\n预算用尽，在所有已选数据上训练至收敛...")
    final_log_path = os.path.join(exp_dir, 'final_convergence_log.txt')
    logger, best_record, _ = get_logfile(args.ckpt_path, args.exp_name, False, None,
                                         log_name=os.path.basename(final_log_path))
    final_train_loader = DataLoader(Subset(train_set_rl, list(train_set_rl.labeled_video_ids)),
                                    batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers)

    _, final_val_acc = train_har_classifier(args, 0, final_train_loader, net_stage3, criterion, optimizer_rl,
                                            val_loader, best_record, logger, scheduler, schedulerP,
                                            final_train=True)
    logger.close()

    print(f"\n--- STAGE 3 COMPLETE --- 收敛后的最终验证集准确率: {final_val_acc:.4f}")
    torch.save(policy_net.state_dict(), os.path.join(exp_dir, 'policy_final.pth'))


if __name__ == '__main__':
    main()