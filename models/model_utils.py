import time
import math
import numpy as np
import os
import random
from scipy.stats import entropy

import torch.nn as nn

from utils.final_utils import get_logfile
from utils.progressbar import progress_bar
from mmaction.apis import init_recognizer
from models.query_network import AdvancedTransformerPolicyNet  # 假设你已保存该类
EPS_START = 0.8
EPS_END = 0.05
EPS_DECAY = 20




def create_models(dataset, model_cfg_path, model_ckpt_path, num_classes,
                  use_policy=True, embed_dim=768):
    """
    创建视频分类网络 + 策略网络（TransformerPolicyNet）
    :param dataset: 数据集名称，用于记录/日志等
    :param model_cfg_path: mmaction2配置文件路径（例如configs/recognition/c3d/c3d_16x1_8x1.py）
    :param model_ckpt_path: 权重文件路径，可以是None或.pth
    :param num_classes: 分类类别数（如HMDB51=51）
    :param use_policy: 是否创建策略网络（用于主动学习）
    :param embed_dim: 视频模型输出的embedding维度
    :return: model, policy_net, target_net
    """
    # Step 1: 初始化视频分类模型（例如C3D、VideoMAE、TSN等）
    model = init_recognizer(
        config=model_cfg_path,
        checkpoint=None,
        device='cuda'
    )
    if model_ckpt_path:
        print(f"Manually loading and fixing checkpoint from: {model_ckpt_path}")
        # 使用 weights_only=True 是更安全的做法
        checkpoint = torch.load(model_ckpt_path, map_location='cpu', weights_only=True)

        # 如果权重在一个 'state_dict' 键下，先取出来
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            # 检查是否是需要添加前缀的 backbone 权重
            if k.startswith('backbone.'):
                new_state_dict[k] = v
                # 为C3D等旧模型添加前缀
            elif not k.startswith('cls_head'):
                new_key = 'backbone.' + k
                new_state_dict[new_key] = v

                # 如果是 cls_head 的权重，则保持原样 (虽然本次加载中不需要)


        # 使用 load_state_dict 加载修复后的权重，strict=False 允许 cls_head 不匹配
        model.load_state_dict(new_state_dict, strict=False)
        print("Checkpoint loaded successfully after fixing keys.")
    print('HAR Backbone model created from MMACTION2.')

    # Step 2: 策略网络（DQN或Transformer）
    if use_policy:
        policy_net = AdvancedTransformerPolicyNet(input_dim=embed_dim).cuda()
        target_net = AdvancedTransformerPolicyNet(input_dim=embed_dim).cuda()
        print(f'Policy/Target network created. Policy net parameters: {count_parameters(policy_net)}')
    else:
        policy_net = None
        target_net = None

    print('All models initialized.\n')
    return model, policy_net, target_net


def count_parameters(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def load_models_for_har(model, load_weights, exp_name_toload, snapshot,
                        exp_name, ckpt_path, checkpointer,
                        policy_net=None, target_net=None,
                        test=False, dataset='hmdb51', use_policy=True,
                        num_classes=51):
    """
    加载 HAR 模型 + 策略网络 + 日志（适配mmaction2 + Transformer policy）

    :param model: mmaction2 视频分类模型（如 C3D）
    :param load_weights: 是否加载预训练权重（来自其他实验）
    :param exp_name_toload: 加载模型的实验名（用于预训练模型）
    :param snapshot: 权重文件名，例如 'ep20.pth'
    :param exp_name: 当前实验名
    :param ckpt_path: 权重文件所在路径
    :param checkpointer: 是否使用 checkpointer 恢复训练
    :param policy_net: Transformer 策略网络
    :param target_net: Transformer 目标网络
    :param test: 是否为测试流程
    :param dataset: 数据集名
    :param use_policy: 是否加载策略网络
    :param num_classes: 类别数（用于日志记录）
    :return: logger, curr_epoch, best_record
    """
    # 1. HAR模型路径
    #print(exp_name_toload,exp_name)
    exp_name_toload=exp_name  #TODO:need to double check
    model_path = os.path.join(ckpt_path, exp_name_toload, snapshot)
    resume_path = os.path.join(ckpt_path, exp_name, snapshot)

    # 2. Policy路径
    policy_path = os.path.join(ckpt_path, exp_name, 'policy_' + snapshot)

    # ---------- 加载 MMACTION2 模型 ----------
    if load_weights and os.path.isfile(model_path):
        print(f'[LOAD] Loading HAR backbone model from {model_path}')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)


    if checkpointer and os.path.isfile(resume_path):
        print(f'[RESUME] Resuming model from {resume_path}')
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

    # ---------- 加载 Policy ----------
    if use_policy and policy_net is not None and os.path.isfile(policy_path):
        print(f'[RESUME] Loading policy net from {policy_path}')
        policy_net.load_state_dict(torch.load(policy_path))
        policy_net.cuda()
        if target_net is not None:
            target_net.load_state_dict(torch.load(policy_path))
            target_net.cuda()

    # ---------- 加载日志 ----------
    logger, best_record, curr_epoch = get_logfile(
        ckpt_path=ckpt_path,
        exp_name=exp_name,
        checkpointer=checkpointer,
        snapshot=snapshot
    )

    return logger, curr_epoch, best_record

def get_region_candidates(candidates, train_set, num_regions=2):
    """Get region candidates function.
    :param candidates: (list) randomly sampled image indexes for images that contain unlabeled regions.
    :param train_set: Training set.
    :param num_regions: Number of regions to take as possible regions to be labeled.
    :return: candidate_regions: List of tuples (int(Image index), int(width_coord), int(height_coord)).
        The coordinate is the left upper corner of the region.
    """
    s = time.time()
    print('Getting region candidates...')
    total_regions = num_regions
    candidate_regions = []
    #### --- Get candidate regions --- ####
    counter_regions = 0
    available_regions = train_set.get_num_unlabeled_regions()
    rx, ry = train_set.get_unlabeled_regions()
    while counter_regions < total_regions and (total_regions - counter_regions) <= available_regions:
        index_ = np.random.choice(len(candidates))
        index = candidates[index_]
        num_regions_left = train_set.get_num_unlabeled_regions_image(int(index))
        if num_regions_left > 0:
            counter_x, counter_y = train_set.get_random_unlabeled_region_image(int(index))
            candidate_regions.append((int(index), counter_x, counter_y))
            available_regions -= 1
            counter_regions += 1
            if num_regions_left == 1:
                candidates.pop(int(index_))
        else:
            print ('This image has no more unlabeled regions!')

    train_set.set_unlabeled_regions(rx, ry)
    print ('Regions candidates indexed! Time elapsed: ' + str(time.time() - s))
    print ('Candidate regions are ' + str(counter_regions))
    return candidate_regions


def get_video_candidates(candidates_idx_list, train_set, num_videos_to_sample):
    """
    获取视频候选列表。
    从当前未标记的视频索引中，随机选择指定数量的视频作为候选。

    :param candidates_idx_list: (list) 当前所有未标记视频的索引列表。
    :param train_set: 训练集对象，用于验证（可选，如果candidates_idx_list已是最新的）
    :param num_videos_to_sample: (int) 要作为候选的视频数量。
    :return: (list) 选中的候选视频索引列表。
    """
    s = time.time()
    print('Getting video candidates...')
    
    # 确保我们不会选择超过实际可用的视频数量
    num_videos_to_sample = min(num_videos_to_sample, len(candidates_idx_list))
    
    if num_videos_to_sample == 0:
        print("No more unlabeled videos to sample.")
        return []

    # 直接从当前的未标记视频索引列表中随机采样
    # 使用 random.sample 确保不重复采样
    selected_candidate_indices = random.sample(candidates_idx_list, num_videos_to_sample)
    
    print(f'Video candidates indexed! Time elapsed: {time.time() - s:.2f}s')
    print(f'Selected {len(selected_candidate_indices)} candidate videos.')
    return selected_candidate_indices


def compute_state(args, model, video_indices, candidate_set, train_set=None):
    #TODO:KL divergence and model feature extraction
    """
    Args:
        args: 参数对象
        model: MMAction2 视频分类模型（VideoMAE等）
        video_indices: List[int]，表示候选未标注视频ID
        candidate_set: 数据集，支持 get_video(idx) 方法返回视频张量
    Returns:
        state: Tensor [N, D]，每个视频的状态特征
        video_indices: 与 state 对齐
    """
    model.eval()
    state = []

    for vid in video_indices:
        # 1. 加载视频 Tensor
        video = candidate_set.get_video(vid)  # shape: [C, T, H, W]
        video = video.unsqueeze(0).cuda()     # [1, C, T, H, W]

        with torch.no_grad():
            logits = model(return_loss=False, imgs=video)  # [1, num_classes]
            probs = F.softmax(logits, dim=1)                # [1, num_classes]
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [1]

        max_prob = torch.max(probs).item()
        entropy_val = entropy.item()
        prob_vec = probs.squeeze(0).cpu()  # shape: [num_classes]

        # 构造特征向量，例如：[entropy, max_prob, class_probabilities]
        feature_vector = torch.cat([torch.tensor([entropy_val, max_prob]), prob_vec], dim=0)
        state.append(feature_vector.unsqueeze(0))

    state = torch.cat(state, dim=0)  # shape: [num_videos, D]
    return state, video_indices


# 文件路径: wcf00317/alrl/alrl-reward_model/models/model_utils.py

# ... (文件内所有现有代码保持不变) ...

# ====================================================================
# ========= 新增一个独立的、专门用于熵选择的函数 =========
# ====================================================================
def select_action_by_entropy(args, entropies):
    """
    一个专门根据熵值来选择样本的独立函数，以确保不影响其他AL算法。

    参数:
    - args: 命令行参数，主要用于获取 num_each_iter。
    - entropies: (list or torch.Tensor) 包含所有候选视频熵值的列表或张量。

    返回:
    - action_indices: (torch.Tensor) 被选中的熵值最高的 k 个样本的索引。
    """
    print('[Entropy] 使用独立的熵选择函数...')
    if not isinstance(entropies, torch.Tensor):
        entropies = torch.tensor(entropies)

    k = args.num_each_iter

    # 使用 torch.topk 找到熵值最高的 k 个样本的索引
    _, action_indices = torch.topk(entropies, k, dim=0)

    return action_indices

# def compute_state_for_har(args, model, train_set, candidate_video_indices, labeled_video_indices=None):
#     """
#     为 HAR 主动学习计算 RL 状态。
#     状态包含来自未标注视频池的特征，以及可选的来自已标注视频子集的特征。
#
#     :param args: 参数对象 (需要包含 num_classes，例如 args.num_classes)
#     :param model: MMAction2 视频分类模型（HAR backbone network）
#     :param candidate_video_indices: List[int]，表示候选未标注视频ID (由 get_video_candidates 返回)
#     :param candidate_set: 数据集，应支持 get_video(idx) 方法返回视频张量。
#                           这里代表的是未标注视频的池。
#     :param labeled_video_indices: List[int]，可选，已标注视频的ID列表，用于计算 policy_net 的 'subset' 输入
#     :return:
#         all_state: dict, 包含 'pool': Tensor [N, D] 未标注视频的状态特征,
#                      'subset': Tensor [M, D] 已标注视频的状态特征 (如果 provided_labeled_indices 不为空)
#         candidate_video_indices: 与 'pool' 对齐的视频索引 (保持原样返回，以便后续关联)
#     """
#     s = time.time()
#     print ('Computing state for HAR active learning...')
#     model.eval() # 确保模型处于评估模式，不计算梯度
#
#     state_pool_features = [] # 存储未标注视频池的状态特征
#
#     # 1. 计算未标注视频池的状态 (即 policy_net 的 'pool' 输入)
#     for vid_idx in candidate_video_indices:
#         # 假设 candidate_set.get_video(vid_idx) 返回预处理好的视频张量 [C, T, H, W]
#         # 这需要在 data/data_utils.py 中的 Dataset 类中实现
#         video_tensor = candidate_set.get_video(vid_idx)
#         # 增加批次维度，并移动到 CUDA
#         video_tensor = video_tensor.unsqueeze(0).cuda() # From [C, T, H, W] to [1, C, T, H, W]
#
#         with torch.no_grad(): # 在特征提取时关闭梯度计算
#             # 调用 HAR 模型进行前向传播，获取分类 logits
#             # MMAction2 模型的 forward 方法：imgs=video_tensor, return_loss=False
#             video_tensor = video_tensor.unsqueeze(dim=1)
#             logits = model(video_tensor, return_loss=False) # [1, num_classes]
#             probs = F.softmax(logits, dim=1)                      # [1, num_classes]
#
#             # 提取状态特征
#             # 方法一：使用分类概率分布的熵和最大概率
#             entropy_val = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item() # 熵值
#             max_prob_val = torch.max(probs).item()                                  # 最大预测概率
#
#             # 方法二：直接使用 logits 或其平坦化版本（如果 PolicyNet 接受）
#             # feature_from_logits = logits.squeeze(0).cpu() # [num_classes]
#
#             # 构造最终的特征向量，例如：[熵, 最大概率, 类别概率分布...]
#             # 确保这里的维度 (D) 与 TransformerPolicyNet 的 embed_dim 匹配
#             # D = 2 (entropy, max_prob) + num_classes
#             feature_vector = torch.cat(
#                 [torch.tensor([entropy_val, max_prob_val], device=probs.device), probs.squeeze(0)], dim=0)
#
#             state_pool_features.append(feature_vector.unsqueeze(0)) # 将 [D] -> [1, D] 并添加到列表中
#
#     # 将所有候选视频的特征拼接成一个大张量
#     if len(state_pool_features) > 0:
#         state_pool_tensor = torch.cat(state_pool_features, dim=0) # [N, D]
#     else:
#         # 如果没有候选视频，返回一个空的张量，维度与预期状态特征维度匹配
#         # 这里假设 D = args.num_classes + 2
#         state_pool_tensor = torch.empty(0, args.num_classes + 2)
#
#     # 2. 计算已标注视频子集的状态 (即 policy_net 的 'subset' 输入)
#     state_subset_features = []
#     # 假设 labeled_video_indices 是一个包含已标注视频 ID 的列表
#     if labeled_video_indices is not None and len(labeled_video_indices) > 0:
#         # 这里需要从 train_set 获取已标注视频数据
#         # 假设 train_set 也有 get_video(idx) 方法来获取视频张量
#         for vid_idx in labeled_video_indices:
#             video_tensor = train_set.get_video(vid_idx) # shape: [C, T, H, W]
#             video_tensor = video_tensor.unsqueeze(0).cuda() # [1, C, T, H, W]
#
#             with torch.no_grad():
#                 logits = model(imgs=video_tensor, return_loss=False)
#                 probs = F.softmax(logits, dim=1)
#                 entropy_val = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()
#                 max_prob_val = torch.max(probs).item()
#                 feature_vector = torch.cat(
#                     [torch.tensor([entropy_val, max_prob_val], device=probs.device), probs.squeeze(0)], dim=0)
#                 state_subset_features.append(feature_vector.unsqueeze(0))
#
#         if len(state_subset_features) > 0:
#             state_subset_tensor = torch.cat(state_subset_features, dim=0) # [M, D]
#         else:
#             state_subset_tensor = torch.empty(0, args.num_classes + 2) # 同样，处理空列表情况
#     else:
#         state_subset_tensor = torch.empty(0, args.num_classes + 2) # 如果没有已标注视频，返回空张量
#
#     # 3. 构建最终的状态字典
#     all_state = {'pool': state_pool_tensor, 'subset': state_subset_tensor}
#
#     print (f'State computed! Time elapsed: {time.time() - s:.2f}s')
#
#     # 原始函数返回 region_candidates，这里我们继续返回 video_indices，以保持调用链一致
#     return all_state, candidate_video_indices # 这里的 candidate_video_indices 现在直接是视频ID列表



import torch
import torch.nn.functional as F
import time


def compute_state_for_har(args, model, train_set, candidate_video_indices, labeled_video_indices=None):
    """
    返回:
      state = {
        'pool':   [N, D]  # 候选池每个样本的 embedding（4096等），已做时空均值
        'subset': [1, D]  # 已标注集的聚合表示（均值）
      }
      candidate_video_indices: 原样返回
      candidate_entropies:     list[float]，每个候选样本的熵
    约定:
      - 仅一次前向: extract_feat -> cls_head -> softmax 计算熵
      - 仅当输入是4维 [C,T,H,W] 才 unsqueeze 到 [1,C,T,H,W]；5维 [N,C,T,H,W] 直接用
      - 全部返回 CPU 张量; 后续在优化器/推入buffer时统一搬到 device
    """
    s = time.time()
    print('Computing state (embeddings + entropy) ...')
    model.eval()

    pool_feats   = []
    pool_entropies = []

    def _forward_one(vid_idx):
        # 取视频张量
        vt = train_set.get_video(vid_idx)  # 形如 [C,T,H,W] 或 [N,C,T,H,W]
        if vt.dim() == 4:   # [C,T,H,W] -> [1,C,T,H,W]
            vt = vt.unsqueeze(0)
        # 若已是 [N,C,T,H,W] 则不动
        vt = vt.cuda(non_blocking=True)

        with torch.no_grad():
            # 一次前向：先拿 embedding
            feats = model.extract_feat(vt)  # 通常返回 list/tuple，或 [N,C',T',H',W']
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            # 时空均值 -> [N, D]
            while feats.dim() > 2:
                feats = feats.mean(dim=-1)
            # 如果前面是 [N, D]，ok；如果是 [D]（少见），补 batch 维
            if feats.dim() == 1:
                feats = feats.unsqueeze(0)

            # logits 来自 cls_head(feats)
            logits = model.cls_head(feats)  # [N, num_classes]
            probs  = F.softmax(logits, dim=1)
            # 每个 clip 的熵
            ent = (-probs * torch.log(probs + 1e-8)).sum(dim=1)  # [N]

            # 如果 get_video 返回了多 clip（N>1），我们按你之前习惯对 N 做均值聚合成单向量
            if feats.size(0) > 1:
                feats = feats.mean(dim=0, keepdim=True)  # [1, D]
                ent   = ent.mean(dim=0, keepdim=True)    # [1]

            return feats.squeeze(0).cpu(), float(ent.item())

    # 1) 候选池
    if candidate_video_indices:
        for vid in candidate_video_indices:
            f, e = _forward_one(vid)
            pool_feats.append(f)       # f: [D] (CPU)
            pool_entropies.append(e)   # float

    if len(pool_feats) > 0:
        pool_tensor = torch.stack(pool_feats, dim=0)  # [N, D] (CPU)
    else:
        pool_tensor = torch.empty(0, args.embed_dim)

    # 2) 已标注集 -> [1,D]
    if labeled_video_indices:
        sub_feats = []
        for vid in labeled_video_indices:
            f, _ = _forward_one(vid)
            sub_feats.append(f)
        subset_tensor = torch.stack(sub_feats, dim=0).mean(dim=0, keepdim=True)  # [1, D]
    else:
        subset_tensor = torch.zeros(1, args.embed_dim)  # [1, D] on CPU

    state = {'pool': pool_tensor, 'subset': subset_tensor}

    print(f'State OK. pool={tuple(state["pool"].shape)}, subset={tuple(state["subset"].shape)}; '
          f'took {time.time()-s:.2f}s')

    return state, candidate_video_indices, pool_entropies
def add_labeled_videos(args, list_existing_videos, videos_to_label_ids, train_set, budget, n_ep):
    """
    此函数将指定视频列表添加到已标注数据集和已存在视频列表中。

    :param args: 参数对象。
    :param list_existing_videos: (list) 所有过去已选定并添加到标注集中的视频索引列表。
    :param videos_to_label_ids: (list) 实际要标记的视频的原始ID列表。
    :param train_set: (torch.utils.data.Dataset) 训练集对象。
    :param budget: (int) 要标注的最大视频数量。
    :param n_ep: (int) 当前的 episode 编号。
    :return: 已存在视频的列表，已更新并包含新视频。
    """
    lab_file_path = os.path.join(args.ckpt_path, args.exp_name, f'labeled_set_ep{n_ep}.txt')
    lab_set = open(lab_file_path, 'a')

    for video_id in videos_to_label_ids:
        if train_set.get_num_labeled_videos() >= budget:
            print(f'Budget reached with {train_set.get_num_labeled_videos()} videos!')
            break

        train_set.add_video_to_labeled(video_id)
        list_existing_videos.append(video_id)
        lab_set.write(f"{video_id}\n")

    lab_set.close()
    print(f'Labeled set has now {train_set.get_num_labeled_videos()} labeled videos.')

    return list_existing_videos


def select_action_for_har(args, policy_net, all_state, steps_done, test=False):
    """
    HAR任务中，根据策略网络/随机/熵选择待标注的clip。

    参数：
    - args: 命令行参数
    - policy_net: Transformer 策略网络，输入clip embedding + 已标注clip embedding
    - all_state: dict, 包含 'pool': [N, D] 待选clip，'subset': [M, D] 已标注clip
    - steps_done: 当前已标注clip数量（用于epsilon-greedy）
    - test: 是否测试阶段（True时强制贪婪选择）

    返回：
    - action: 待选 clip 的索引 (tensor)
    - steps_done: 更新后的步数
    - ent: 若使用熵策略，返回选中clip的熵；否则为0
    """
    state_pool = all_state['pool']         # [N, D]
    state_subset = all_state['subset']     # [M, D]
    ent = 0

    if args.al_algorithm == 'dqn':
        policy_net.eval()
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1

        if sample > eps_threshold or test:

            print('[DQN] Using policy network to select clip...')
            with torch.no_grad():
                q_vals = []
                batch_size = 16
                for i in range(0, state_pool.size(0), batch_size):
                    clip_batch = state_pool[i:i + batch_size].cuda()  # [B, D]
                    # repeat subset embedding 为 clip_batch 的 batch size
                    if state_subset.dim() == 2:
                        subset_batch = state_subset.unsqueeze(0).repeat(clip_batch.size(0), 1, 1).cuda()  # [B,1,D]
                    else:
                        subset_batch = state_subset.repeat(clip_batch.size(0), 1, 1).cuda() 
                    # subset_batch = state_subset.unsqueeze(0).repeat(clip_batch.size(0), 1, 1).cuda()  # [B, M, D]
                    # print("--- Ablation Study: Historical information (subset) is zeroed out before policy net. ---")
                    # subset_batch = torch.zeros_like(subset_batch) # todo: ablation std
                    q_val = policy_net(clip_batch, subset_batch).cpu()  # 输出 [B, 1]
                    q_vals.append(q_val)
                q_vals = torch.cat(q_vals, dim=0).squeeze()  # 从 [N, 1] 变为 [N]
                # --- 开始修改 ---
                k = args.num_each_iter
                # 使用 topk 找到 Q 值最高的 k 个动作的索引
                action = torch.topk(q_vals, k, dim=0)[1]  # [1] 代表我们只关心索引
        else:
            print('[DQN] Random exploration')
            action = torch.randperm(state_pool.size(0))[:args.num_each_iter]
    elif args.al_algorithm == 'random':
        action = torch.randperm(state_pool.size(0))[:args.num_each_iter]
    elif args.al_algorithm == 'entropy':
        # 你需要提前将 logits 存入 state_pool（假设shape为 [N, C]）
        probs = state_pool[:, 2:]  # 剔除 entropy, max_prob
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        # probs = F.softmax(state_pool, dim=-1)         # [N, C]
        # log_probs = F.log_softmax(state_pool, dim=-1)
        # entropy = -torch.sum(probs * log_probs, dim=-1)  # [N]
        k = args.num_each_iter
        ent, action = torch.topk(entropy, k, dim=0)  # ent是最高的k个熵值，action是索引
    else:
        raise ValueError(f"[select_action_for_har] Unknown algorithm: {args.al_algorithm}")

    return action, steps_done, ent

def add_labeled_images(args, list_existing_images, region_candidates, train_set, action_list, budget, n_ep):
    """This function adds an image, indicated by 'action_list' out of 'region_candidates' list
     and adds it into the labeled dataset and the list of existing images.

    :(argparse.ArgumentParser) args: The parser with all the defined arguments.
    :param list_existing_images: (list) of tuples (image idx, region_x, region_y) of all regions that have
            been selected in the past to add them to the labeled set.
    :param region_candidates: (list) List of all possible regions to add.
    :param train_set: (torch.utils.data.Dataset) Training set.
    :param action_list: Selected indexes of the regions in 'region_candidates' to be labeled.
    :param budget: (int) Number of maximum regions we want to label.
    :param n_ep: (int) Number of episode.
    :return: List of existing images, updated with the new image.
    """

    lab_set = open(os.path.join(args.ckpt_path, args.exp_name, 'labeled_set_' + str(n_ep) + '.txt'), 'a')
    for i, action in enumerate(action_list):
        if train_set.get_num_labeled_regions() >= budget:
            print ('Budget reached with ' + str(train_set.get_num_labeled_regions()) + ' regions!')
            break
        im_toadd = region_candidates[i, action, 0]
        train_set.add_indice(im_toadd, (region_candidates[i, action, 1], region_candidates[i, action, 2]))
        list_existing_images.append(tuple(region_candidates[i, action]))
        lab_set.write("%i,%i,%i" % (
            im_toadd, region_candidates[i, action, 1], region_candidates[i, action, 2]))
        lab_set.write("\n")
    print('Labeled set has now ' + str(train_set.get_num_labeled_regions()) + ' labeled regions.')

    return list_existing_images


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def compute_bald(predictions):
    ### Compute BALD ###
    expected_entropy = - torch.mean(torch.sum(predictions * torch.log(predictions + 1e-10), dim=1),
                                    dim=0)
    expected_p = torch.mean(predictions, dim=0)  # [batch_size, n_classes]
    pred_py = expected_p.max(0)[1]
    entropy_expected_p = - torch.sum(expected_p * torch.log(expected_p + 1e-10),
                                     dim=0)  # [batch size]
    bald_acq = entropy_expected_p - expected_entropy
    return bald_acq.unsqueeze(0), pred_py.unsqueeze(0)


def add_kl_pool2(state, n_cl=19):
    sim_matrix = torch.zeros((state.shape[0], state.shape[1], 32))
    all_cand = state[:, :, 0:n_cl + 1].view(-1, n_cl + 1).transpose(1, 0)
    for i in range(state.shape[0]):
        pool_hist = state[i, :, 0:n_cl + 1]
        for j in range(pool_hist.shape[0]):
            prov_sim = entropy(pool_hist[j:j + 1].repeat(all_cand.shape[1], 1).transpose(0, 1), all_cand)
            hist, _ = np.histogram(prov_sim, bins=32)
            hist = hist / hist.sum()
            sim_matrix[i, j, :] = torch.Tensor(hist)
    state = torch.cat([state, sim_matrix], dim=2)
    return state


def create_feature_vector_3H_region_kl_sim(pred_region, ent_region, train_set, num_classes=19, reg_sz=(128, 128)):
    unique, counts = np.unique(pred_region, return_counts=True)
    sample_stats = np.zeros(num_classes + 1) + 1E-7
    sample_stats[unique.astype(int)] = counts
    sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        sample_stats += (5 - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
            -1)).tolist()  # min entropy
        sample_stats += F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        sample_stats += F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
    if len(train_set.balance_cl) > 0:
        inp_hist = sample_stats[0:num_classes + 1]
        sim_sample = entropy(np.repeat(np.asarray(inp_hist)[:, np.newaxis], len(train_set.balance_cl), axis=1),
                             np.asarray(train_set.balance_cl).transpose(1, 0))
        hist, _ = np.histogram(sim_sample, bins=32)
        sim_lab = list(hist / hist.sum())
        sample_stats += sim_lab
    else:
        sample_stats += [0.0] * 32
    return sample_stats


def create_feature_vector_3H_region_kl(pred_region, ent_region, num_classes=19, reg_sz=(128, 128)):
    unique, counts = np.unique(pred_region, return_counts=True)
    sample_stats = np.zeros(num_classes + 1) + 1E-7
    sample_stats[unique.astype(int)] = counts
    sample_stats = sample_stats.tolist()
    sz = ent_region.size()
    ks_x = int(reg_sz[0] // 8)
    ks_y = int(reg_sz[1] // 8)
    with torch.no_grad():
        sample_stats += (5 - F.max_pool2d(5 - ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(
            -1)).tolist()  # min entropy
        sample_stats += F.avg_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
        sample_stats += F.max_pool2d(ent_region.view(1, 1, sz[0], sz[1]), kernel_size=(ks_y, ks_x)).view(-1).tolist()
    return sample_stats


def compute_entropy_seg(args, im_t, net):
    '''
    Compute entropy function
    :param args:
    :param im_t:
    :param net:
    :return:
    '''
    net.eval()
    if im_t.dim() == 3:
        im_t_sz = im_t.size()
        im_t = im_t.view(1, im_t_sz[0], im_t_sz[1], im_t_sz[2])

    out, _ = net(im_t)
    out_soft_log = F.log_softmax(out)
    out_soft = torch.exp(out_soft_log)
    ent = - torch.sum(out_soft * out_soft_log, dim=1).detach().cpu()  # .data.numpy()
    del (out)
    del (out_soft_log)
    del (out_soft)
    del (im_t)

    return ent


import torch
import torch.nn.functional as F

def optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP,
                        GAMMA, BATCH_SIZE,TAU, grad_clip=1.0, use_padding=True):
    import torch
    import torch.nn.functional as F
    import random

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(memory) < BATCH_SIZE:
        return

    # -------------------- Sample a minibatch --------------------
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # ---------- Build current (s,a) batch ----------
    # state_pool: 每条是“被选样本的embedding”，形状可能是 [D] 或 [1,D]
    if torch.is_tensor(batch.state_pool[0]) and batch.state_pool[0].dim() == 1:
        state_batch_pool = torch.stack(batch.state_pool, dim=0).to(device)   # [B, D]
    else:
        # 兼容 [1,D] 的情况
        state_batch_pool = torch.cat(batch.state_pool, dim=0).to(device)     # [B,1,D] or [B,D]
        if state_batch_pool.dim() == 3 and state_batch_pool.size(1) == 1:
            state_batch_pool = state_batch_pool.squeeze(1)                   # -> [B, D]
    assert state_batch_pool.dim() == 2, f"state_batch_pool should be [B,D], got {state_batch_pool.shape}"

    # state_subset: 每条是 [1,M,D] 或 [M,D]
    state_batch_subset = torch.cat(batch.state_subset, dim=0).to(device)     # [B,1,M,D] or [B,M,D] or [B,D]
    while state_batch_subset.dim() > 3:
        state_batch_subset = state_batch_subset.squeeze(1)                   # 去掉多余的1维
    if state_batch_subset.dim() == 2:
        state_batch_subset = state_batch_subset.unsqueeze(1)                 # [B,1,D] -> [B,1,D] (容错)
    assert state_batch_subset.dim() == 3, f"state_batch_subset must be [B,M,D], got {state_batch_subset.shape}"

    # action：当前实现未用到，但保留以便debug
    action_batch = torch.stack(
        [a if torch.is_tensor(a) else torch.tensor(a) for a in batch.action], dim=0
    ).to(device)  # [B]

    # reward：推的是标量（0-D）就行，这里统一成 [B]
    reward_elems = []
    for r in batch.reward:
        if torch.is_tensor(r):
            reward_elems.append(r.reshape(()))  # 保证0-D
        else:
            reward_elems.append(torch.tensor(r, dtype=torch.float))
    reward_batch = torch.stack(reward_elems, dim=0).to(device)  # [B]
    reward_mean = reward_batch.mean()
    reward_std = reward_batch.std()
    reward_batch = (reward_batch - reward_mean) / (reward_std + 1e-7)  # 加上 epsilon
    print(f"[DEBUG] Normalized Rewards -> Mean: {reward_batch.mean().item():.6f}, Std: {reward_batch.std().item():.6f}")
    # -------------------- Q(s,a) --------------------
    # policy_net 接口： (x=[B,D], subset=[B,M,D]) -> [B] or [B,1]
    q_sa = policy_net(state_batch_pool, state_batch_subset)
    if q_sa.dim() == 2 and q_sa.size(1) == 1:
        q_sa = q_sa.squeeze(1)
    q_sa = torch.nan_to_num(q_sa, nan=0.0, posinf=1e6, neginf=-1e6)  # 数值保护
    assert q_sa.shape == reward_batch.shape, f"q_sa shape {q_sa.shape} vs reward {reward_batch.shape}"

    # -------------------- target: max_{a'} Q_target(s',a') --------------------
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # mask：哪些transition有下一个状态
    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state_pool),
                                  device=device, dtype=torch.bool)
    if non_final_mask.any():
        # 收集非终止样本的 next_subset / next_pool
        nf_next_subset_list = [s for s in batch.next_state_subset if s is not None]
        nf_next_pool_list   = [p for p in batch.next_state_pool   if p is not None]

        # next_subset -> [B_nf, M, D]
        nf_next_subset = torch.cat(nf_next_subset_list, dim=0).to(device)
        while nf_next_subset.dim() > 3:
            nf_next_subset = nf_next_subset.squeeze(1)
        if nf_next_subset.dim() == 2:
            nf_next_subset = nf_next_subset.unsqueeze(0)  # [1,M,D] -> [1,M,D]
        assert nf_next_subset.dim() == 3, f"nf_next_subset must be [B_nf,M,D], got {nf_next_subset.shape}"

        # 我们希望 next_pool 统一为 [B_nf, K, D]；K 可变时用padding或逐样本fallback
        try:
            ks = []
            normed = []
            for p in nf_next_pool_list:
                x = p.to(device)
                # 兼容 [1,K,D] / [K,D] / [1,1,D] / [D]
                if x.dim() == 4 and x.size(1) == 1:
                    x = x.squeeze(1)            # [1,K,D] -> [K,D]
                if x.dim() == 1:
                    x = x.unsqueeze(0)          # [D] -> [1,D]
                if x.dim() == 2 and x.size(0) == 1:
                    # [1,D] 视作单动作
                    k_i = 1
                    x = x.unsqueeze(1)          # -> [1,1,D]
                elif x.dim() == 2:
                    # [K,D]
                    k_i = x.size(0)
                    x = x.unsqueeze(0)          # -> [1,K,D]
                elif x.dim() == 3:
                    # [1,K,D]
                    k_i = x.size(1)
                else:
                    raise RuntimeError(f"Unexpected next_pool shape: {tuple(x.shape)}")
                ks.append(k_i)
                normed.append(x)

            max_k = max(ks)
            if len(set(ks)) != 1 and not use_padding:
                raise RuntimeError(f"Variable K={ks} and use_padding=False")

            # padding到同一K
            pads = []
            for x, k_i in zip(normed, ks):
                if use_padding and k_i < max_k:
                    pad_k = max_k - k_i
                    x = F.pad(x, (0, 0, 0, pad_k))  # pad K 维
                pads.append(x)
            nf_next_pool = torch.cat(pads, dim=0)            # [B_nf, K, D]

            # 批量算 Q_target(s', a')，并在K上取max
            B_nf, K, D = nf_next_pool.shape
            flat_next_x = nf_next_pool.reshape(B_nf * K, D)  # [B_nf*K, D]
            rep_next_subset = nf_next_subset.unsqueeze(1).expand(B_nf, K, *nf_next_subset.shape[1:]) \
                                               .reshape(B_nf * K, *nf_next_subset.shape[1:])
            with torch.no_grad():
                q_next_policy = policy_net(flat_next_x, rep_next_subset).reshape(B_nf, K)
                q_next_policy = torch.nan_to_num(q_next_policy, nan=float('-inf'))  # Treat NaN as worst action
                if use_padding and len(set(ks)) != 1:  # Apply mask before argmax
                    valid_mask = torch.arange(K, device=device).unsqueeze(0) < torch.tensor(ks,
                                                                                            device=device).unsqueeze(1)
                    q_next_policy = q_next_policy.masked_fill(~valid_mask, float('-inf'))
                best_next_actions = q_next_policy.argmax(dim=1, keepdim=True)  # Shape: [B_nf, 1]

                # b) 使用 target_net 评估这些被选定动作的Q值
                q_next_target = target_net(flat_next_x, rep_next_subset).reshape(B_nf, K)
                q_next_target = torch.nan_to_num(q_next_target, nan=0.0, posinf=1e6,
                                                 neginf=-1e6)  # NaN handling for value
                # 使用 gather() 来挑选出与 best_next_actions 对应的 Q 值
                q_next_max = q_next_target.gather(1, best_next_actions).squeeze()  # Shape: [B_nf]

                # (移除旧的 q_next_all 计算和 mask 应用)
                next_state_values[non_final_mask] = q_next_max

        except RuntimeError:
            # 逐样本fallback（K可变时）
            q_next_max_list = []
            with torch.no_grad():
                for pool_j, subset_j in zip(nf_next_pool_list, nf_next_subset_list):
                    pj = pool_j.to(device)
                    sj = subset_j.to(device)
                    while sj.dim() > 3:
                        sj = sj.squeeze(1)
                    if sj.dim() == 2:
                        sj = sj.unsqueeze(0)  # [1,M,D]

                    if pj.dim() == 4 and pj.size(1) == 1:
                        pj = pj.squeeze(1)   # [1,K,D] -> [K,D]
                    if pj.dim() == 3 and pj.size(0) == 1:
                        pj = pj.squeeze(0)   # [1,K,D] -> [K,D]
                    if pj.dim() == 1:
                        pj = pj.unsqueeze(0) # [D] -> [1,D]

                    Kj = pj.size(0) if pj.dim() == 2 else 1
                    rep_sj = sj.expand(Kj, *sj.shape[1:])  # [Kj,M,D]
                    q_all_j = target_net(pj, rep_sj).squeeze(-1)  # [Kj]
                    q_all_j = torch.nan_to_num(q_all_j, nan=0.0, posinf=1e6, neginf=-1e6)
                    q_next_max_list.append(q_all_j.max())
            next_state_values[non_final_mask] = torch.stack(q_next_max_list, dim=0)

    # -------------------- Bellman target & loss --------------------
    target = reward_batch + GAMMA * next_state_values
    target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)

    loss = F.smooth_l1_loss(q_sa, target)

    # -------------------- Backprop --------------------
    optimizerP.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=grad_clip)
    optimizerP.step()
    #实现软更新
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        # target_weights = (1-TAU)*target_weights + TAU*policy_weights
        target_net_state_dict[key] = target_net_state_dict[key] * (1 - TAU) + policy_net_state_dict[key] * TAU
    target_net.load_state_dict(target_net_state_dict)

    # for logging
    optimize_model_conv.last_loss = float(loss.item())
    optimize_model_conv.last_q_mean = float(q_sa.mean().item())
    optimize_model_conv.last_reward_mean = float(reward_batch.mean().item())


def debug_optimize_model_conv(memory, Transition, BATCH_SIZE=4):
    import torch
    import random

    print("\n=== [DEBUG MODE] Checking shapes inside ReplayMemory ===")

    if len(memory) < BATCH_SIZE:
        print(f"[WARN] Memory not enough: len={len(memory)}, need {BATCH_SIZE}")
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    print(f"[INFO] Sampled {BATCH_SIZE} transitions.")
    print("-----------------------------------------------------------")

    # 检查每个字段的类型与形状
    for i in range(BATCH_SIZE):
        print(f"\n[Transition #{i}] -----------------------------")

        # state_pool
        s_pool = batch.state_pool[i]
        if s_pool is None:
            print("state_pool: None")
        elif torch.is_tensor(s_pool):
            print(f"state_pool: tensor, shape={tuple(s_pool.shape)}, dtype={s_pool.dtype}")
        else:
            print(f"state_pool: {type(s_pool)}")

        # state_subset
        s_sub = batch.state_subset[i]
        if s_sub is None:
            print("state_subset: None")
        elif torch.is_tensor(s_sub):
            print(f"state_subset: tensor, shape={tuple(s_sub.shape)}, dtype={s_sub.dtype}")
        else:
            print(f"state_subset: {type(s_sub)}")

        # action
        act = batch.action[i]
        if torch.is_tensor(act):
            print(f"action: tensor, shape={tuple(act.shape)}, value range=({act.min().item():.3g},{act.max().item():.3g})")
        else:
            print(f"action: {act} ({type(act)})")

        # next_state_pool
        nxtp = batch.next_state_pool[i]
        if nxtp is None:
            print("next_state_pool: None (terminal)")
        elif torch.is_tensor(nxtp):
            print(f"next_state_pool: tensor, shape={tuple(nxtp.shape)}, dtype={nxtp.dtype}")
        else:
            print(f"next_state_pool: {type(nxtp)}")

        # next_state_subset
        nxts = batch.next_state_subset[i]
        if nxts is None:
            print("next_state_subset: None")
        elif torch.is_tensor(nxts):
            print(f"next_state_subset: tensor, shape={tuple(nxts.shape)}, dtype={nxts.dtype}")
        else:
            print(f"next_state_subset: {type(nxts)}")

        # reward
        r = batch.reward[i]
        if torch.is_tensor(r):
            print(f"reward: tensor, shape={tuple(r.shape)}, val={r.item():.6f}")
        else:
            print(f"reward: {r} ({type(r)})")

    # 汇总统计
    shapes_pool = [tuple(s.shape) for s in batch.next_state_pool if torch.is_tensor(s)]
    if len(shapes_pool) > 0:
        unique_shapes = list(set(shapes_pool))
        print("\n[SUMMARY] Unique shapes of next_state_pool in this batch:", unique_shapes)
    else:
        print("\n[SUMMARY] No valid tensor in next_state_pool.")

    print("=== [DEBUG MODE] Done ===\n")
