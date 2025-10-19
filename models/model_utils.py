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
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200




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
    【熵奖励修正版】
    此版本同时计算4096维特征嵌入（用于状态）和每个视频的熵（用于奖励模型训练数据），
    并作为三个独立的变量返回，以匹配 run.py 中的调用。
    """
    s = time.time()
    print('Computing state (4096-dim embeddings) AND entropy for reward calculation...')
    model.eval()

    # --- 新增：初始化两个列表，分别存储特征和熵 ---
    state_pool_features = []
    candidate_entropies = []

    # --------------------------------------------------------------------
    # 内部辅助函数现在需要同时返回特征和熵
    def _get_feature_and_entropy_for_single_video(vid_idx):
        video_tensor = train_set.get_video(vid_idx)
        video_tensor = video_tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            # 1. 为了计算熵，我们仍然需要做一次完整的分类前向传播来获取概率
            # print(video_tensor.shape)
            logits = model(video_tensor, return_loss=False)
            logits = model.cls_head(logits)
            if logits.shape[0] > 1:
                logits = logits.mean(dim=0, keepdim=True)

            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()

            # 2. 获取4096维特征用于状态表示
            features = model.extract_feat(video_tensor)[0]
            if features.shape[0] > 1:
                features = features.mean(dim=0, keepdim=True)

            if features.ndim == 5:
                features = features.mean(dim=(2, 3, 4))
            elif features.ndim == 4:
                features = features.mean(dim=(2, 3))
            elif features.ndim == 3:
                features = features.mean(dim=2)

            if features.dim() == 1:
                features = features.unsqueeze(0)

            # 同时返回特征和熵
            return features, entropy

    # --------------------------------------------------------------------

    # 1. 计算未标注视频池的状态和熵
    if candidate_video_indices:
        for vid_idx in candidate_video_indices:
            # 接收特征和熵两个返回值
            feature, entropy = _get_feature_and_entropy_for_single_video(vid_idx)
            state_pool_features.append(feature)
            # 将熵存入列表
            candidate_entropies.append(entropy)

    state_pool_tensor = torch.cat(state_pool_features, dim=0) if state_pool_features else torch.empty(0, args.embed_dim)

    # 2. 计算已标注视频子集的状态 (这部分逻辑不变，我们只需要特征)
    if labeled_video_indices:
        state_subset_features = []
        for vid_idx in labeled_video_indices:
            # 在这里我们只需要特征，可以忽略熵的返回值
            feature, _ = _get_feature_and_entropy_for_single_video(vid_idx)
            state_subset_features.append(feature)

        all_subset_features = torch.cat(state_subset_features, dim=0)
        fixed_size_subset_tensor = torch.mean(all_subset_features, dim=0, keepdim=True)
    else:
        fixed_size_subset_tensor = torch.zeros(1, args.embed_dim, device='cuda')

    # 3. 构建最终的状态字典
    all_state = {'pool': state_pool_tensor.cpu(), 'subset': fixed_size_subset_tensor.cpu()}

    print(f'State and entropies computed! Pool shape: {all_state["pool"].shape}. Time: {time.time() - s:.2f}s')

    # --- 核心修改：返回三个值 ---
    return all_state, candidate_video_indices, candidate_entropies

# def compute_state_for_har(args, model, train_set, candidate_video_indices, labeled_video_indices=None):
#     """
#     计算RL状态的【逐一处理修正版】。
#     此版本修复了参数和设备bug，但为了调试方便，恢复了逐一处理视频的逻辑。
#     注意：此版本运行速度会比批处理版本慢很多。
#
#     :param args: 参数对象
#     :param model: MMAction2 视频分类模型
#     :param train_set: 完整的数据集对象
#     :param candidate_video_indices: List[int]，候选未标注视频ID。
#     :param labeled_video_indices: List[int]，可选，已标注视频的ID列表。
#     :return:
#         all_state: dict, 包含 'pool' 和 'subset' 的状态特征张量。
#         candidate_video_indices: 原样返回。
#     """
#     s = time.time()
#     print('Computing state (single item processing for debugging)...')
#     model.eval()
#
#
#     # --------------------------------------------------------------------
#     # 为了避免代码重复，我们先定义一个处理单个视频的内部辅助函数
#     def _get_feature_for_single_video(vid_idx):
#         # 统一使用 train_set 的 get_video 方法，它应该返回一个简单的 [C, T, H, W] 张量
#         video_tensor = train_set.get_video(vid_idx)
#         # 增加批次维度(B=1)，并移动到 CUDA
#         video_tensor = video_tensor.unsqueeze(0).cuda()
#
#         with torch.no_grad():
#             # 为模型准备输入
#             # video_tensor = video_tensor.unsqueeze(1)
#             logits = model(video_tensor, return_loss=False)
#             if logits.shape[0] > 1:  # 如果有多个clips
#                 # 在进行softmax之前，先对logits求平均
#                 logits = logits.mean(dim=0, keepdim=True)  # 形状变为 [1, num_classes]
#
#             probs = F.softmax(logits, dim=1)
#
#             # 特征计算
#             entropy_val = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).item()
#             max_prob_val = torch.max(probs).item()
#
#             # 拼接特征向量 (所有张量都在GPU上，无设备冲突)
#             # print(entropy_val.shape, max_prob_val.shape)
#             feature_vector = probs.squeeze(0)#torch.cat(
#                 #[torch.tensor([entropy_val, max_prob_val], device=probs.device), probs.squeeze(0)], dim=0)
#             # print(feature_vector.shape)
#             # print("=================")
#             # 返回一个 [1, D] 形状的张量，方便后续拼接
#             return feature_vector.unsqueeze(0)
#
#     # --------------------------------------------------------------------
#
#     # 1. 计算未标注视频池的状态
#     state_pool_features = []
#     if candidate_video_indices:
#         for vid_idx in candidate_video_indices:
#             feature = _get_feature_for_single_video(vid_idx)
#             state_pool_features.append(feature)
#
#     state_pool_tensor = torch.cat(state_pool_features, dim=0) if state_pool_features else torch.empty(0,
#                                                                                                       args.num_classes + 2)
#
#     # 2. 计算已标注视频子集的状态
#     if labeled_video_indices:
#         state_subset_features = []
#         for vid_idx in labeled_video_indices:
#             feature = _get_feature_for_single_video(vid_idx)
#             state_subset_features.append(feature)
#
#         # ✅ 首先，像原来一样拼接成一个 [num_labeled, num_classes] 的大张量
#         all_subset_features = torch.cat(state_subset_features, dim=0)
#
#         # ✅ 然后，对所有已标注视频的特征取平均值，聚合成一个单一的向量
#         # keepdim=True 确保输出形状是 [1, num_classes] 而不是 [num_classes]
#         fixed_size_subset_tensor = torch.mean(all_subset_features, dim=0, keepdim=True)
#     else:
#         # 如果没有任何已标注视频，返回一个正确形状的零向量
#         fixed_size_subset_tensor = torch.zeros(1, args.num_classes, device='cuda')
#
#     # state_subset_tensor = torch.cat(state_subset_features, dim=0) if state_subset_features else torch.empty(0,
#     #                                                                                                        args.num_classes + 2)
#
#     # 3. 构建最终的状态字典
#     # all_state = {'pool': state_pool_tensor.cpu(), 'subset': state_subset_tensor.cpu()}
#     all_state = {'pool': state_pool_tensor.cpu(), 'subset': fixed_size_subset_tensor.cpu()}
#
#     print(f'State computed! Pool shape: {all_state["pool"].shape}, Subset shape: {all_state["subset"].shape}. Time: {time.time() - s:.2f}s')
#
#
#     return all_state, candidate_video_indices

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
                        GAMMA, BATCH_SIZE, grad_clip=1.0, use_padding=True):
    import torch
    import torch.nn.functional as F
    import random

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n=== [DEBUG] Enter optimize_model_conv ===")
    print(f"len(memory): {len(memory)}, BATCH_SIZE: {BATCH_SIZE}")

    if len(memory) < BATCH_SIZE:
        print("[DEBUG] Memory not enough, skipping update.")
        return

    # ---- Sample ----
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    if random.random() < 0.02:
        print("[DEBUG] Sample example:]")
        ex_r = batch.reward[0].item() if torch.is_tensor(batch.reward[0]) else batch.reward[0]
        print("  reward:", ex_r)
        print("  action:", batch.action[0])
        print("  state_pool shape:", tuple(batch.state_pool[0].shape) if torch.is_tensor(batch.state_pool[0]) else None)
        print("  next_state_pool is None?", batch.next_state_pool[0] is None)

    # ---- Masks ----
    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state_pool),
                                  device=device, dtype=torch.bool)
    print("[DEBUG] non_final_mask shape:", tuple(non_final_mask.shape),
          "sum:", int(non_final_mask.sum().item()))

    # ---- Batches ----
    state_batch_pool   = torch.cat(batch.state_pool).to(device)         # [B,D] or [B,1,D]
    state_batch_subset = torch.cat(batch.state_subset).to(device)       # [B,M,D] or [B,1,1,D] or [B,D]

    # x -> [B,D]
    if state_batch_pool.dim() == 3 and state_batch_pool.size(1) == 1:
        state_batch_pool = state_batch_pool.squeeze(1)
    assert state_batch_pool.dim() == 2, f"state_batch_pool should be [B,D], got {state_batch_pool.shape}"

    # subset -> [B,M,D]
    while state_batch_subset.dim() > 3:
        state_batch_subset = state_batch_subset.squeeze(1)
    if state_batch_subset.dim() == 2:
        state_batch_subset = state_batch_subset.unsqueeze(1)
    elif state_batch_subset.dim() == 3 and state_batch_subset.size(1) == 0:
        state_batch_subset = state_batch_subset.reshape(state_batch_subset.size(0), 1, state_batch_subset.size(-1))
    assert state_batch_subset.dim() == 3, f"state_batch_subset should be [B,M,D], got {state_batch_subset.shape}"

    action_batch = torch.cat([a.unsqueeze(0) for a in batch.action]).to(device)  # [B]
    reward_batch = torch.cat([r.unsqueeze(0) for r in batch.reward]).to(device)  # [B]

    print("[DEBUG] state_batch_pool:", tuple(state_batch_pool.shape))
    print("[DEBUG] state_batch_subset:", tuple(state_batch_subset.shape))
    print("[DEBUG] action_batch:", tuple(action_batch.shape),
          "min:", float(action_batch.min().item()), "max:", float(action_batch.max().item()))
    r_min, r_max, r_std = float(reward_batch.min().item()), float(reward_batch.max().item()), float(reward_batch.std().item())
    print("[DEBUG] reward_batch:", tuple(reward_batch.shape), "min:", r_min, "max:", r_max, "std:", r_std)
    if r_std < 1e-6:
        print("[WARN] Reward variance too small -> 模型很难学到差异。请检查 reward 设计。")

    # ---- Q(s,a) ----
    q_sa = policy_net(state_batch_pool, state_batch_subset)
    if q_sa.dim() == 2 and q_sa.size(1) == 1:
        q_sa = q_sa.squeeze(1)
    q_sa = torch.nan_to_num(q_sa, nan=0.0, posinf=1e6, neginf=-1e6)
    print("[DEBUG] q_sa shape:", tuple(q_sa.shape),
          "min:", float(q_sa.min().item()), "max:", float(q_sa.max().item()))

    # ---- target: max_{a'} Q_target(s',a') ----
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    if non_final_mask.any():
        nf_next_subset_list = [s for s in batch.next_state_subset if s is not None]
        nf_next_pool_list   = [p for p in batch.next_state_pool   if p is not None]

        # subset_nf -> [B_nf,M,D]
        nf_next_subset = torch.cat(nf_next_subset_list).to(device)
        while nf_next_subset.dim() > 3:
            nf_next_subset = nf_next_subset.squeeze(1)
        if nf_next_subset.dim() == 2:
            nf_next_subset = nf_next_subset.unsqueeze(1)
        assert nf_next_subset.dim() == 3, f"nf_next_subset must be [B_nf,M,D], got {nf_next_subset.shape}"

        # 尝试批量路径：要求所有样本的 K 相同
        try:
            # 预检查：打印每个样本的 K 以便诊断
            ks = []
            for idx, p in enumerate(nf_next_pool_list):
                shape = tuple(p.shape)
                # 兼容 [1,K,D] / [K,D] / [1,1,K,D]
                if len(shape) == 4 and shape[1] == 1:
                    k_i = shape[2]
                elif len(shape) == 3:
                    k_i = shape[1]
                elif len(shape) == 2:
                    k_i = 1
                else:
                    raise RuntimeError(f"Unexpected next_state_pool item shape: {shape}")
                ks.append(k_i)
            if len(set(ks)) != 1 and not use_padding:
                raise RuntimeError(f"Variable K in batch (ks={ks}) and use_padding=False")

            # 统一到 [B_nf,K,D]
            normed = []
            max_k = max(ks)
            for p in nf_next_pool_list:
                x = p.to(device)
                if x.dim() == 4 and x.size(1) == 1:
                    x = x.squeeze(1)               # [1,K,D] -> [K,D]
                if x.dim() == 2:
                    x = x.unsqueeze(0)             # [D] or [K,D]? if [D], becomes [1,D] then next line handles
                if x.dim() == 2 and x.size(0) != 1:  # [K,D] already; make it [1,K,D]
                    x = x.unsqueeze(0)
                if x.dim() == 2 and x.size(0) == 1:  # [1,D] -> [1,1,D]
                    x = x.unsqueeze(1)
                if x.dim() == 3 and x.size(1) == 0:
                    x = x.reshape(1, 1, x.size(-1))
                # padding（可选）
                if use_padding and x.size(1) < max_k:
                    pad_k = max_k - x.size(1)
                    x = F.pad(x, (0, 0, 0, pad_k))  # pad K dimension with zeros
                normed.append(x)  # [1,K,D]
            nf_next_pool = torch.cat(normed, dim=0)  # [B_nf,K,D]

            B_nf, K, D = nf_next_pool.shape
            flat_next_x = nf_next_pool.reshape(B_nf * K, D)
            rep_next_subset = nf_next_subset.unsqueeze(1).expand(B_nf, K, *nf_next_subset.shape[1:]) \
                                               .reshape(B_nf * K, *nf_next_subset.shape[1:])
            with torch.no_grad():
                q_next_all = target_net(flat_next_x, rep_next_subset).reshape(B_nf, K)
                q_next_all = torch.nan_to_num(q_next_all, nan=0.0, posinf=1e6, neginf=-1e6)
                # 对 padding 的 0 向量（若启用 padding）做屏蔽：置为 -inf
                if use_padding and len(set(ks)) != 1:
                    valid_mask = torch.arange(K, device=device).unsqueeze(0) < torch.tensor(ks, device=device).unsqueeze(1)
                    q_next_all = q_next_all.masked_fill(~valid_mask, float('-inf'))
                q_next_max = q_next_all.max(dim=1).values

            next_state_values[non_final_mask] = q_next_max
            print("[DEBUG] q_next_all:", tuple(q_next_all.shape),
                  "q_next_max min/max:", float(q_next_max.min().item()), float(q_next_max.max().item()))

        except RuntimeError as e:
            # 逐样本安全路径（可变 K）
            print("[WARN] Batch has variable K or shape mismatch; fallback per-sample. err:", e)
            with torch.no_grad():
                # 注意：这里的顺序与 nf_next_pool_list 一一对应
                q_next_max_list = []
                for j, (pool_j, subset_j) in enumerate(zip(nf_next_pool_list, nf_next_subset_list)):
                    pj = pool_j.to(device)
                    sj = subset_j.to(device)
                    # subset -> [1,M,D]
                    while sj.dim() > 3:
                        sj = sj.squeeze(1)
                    if sj.dim() == 2:
                        sj = sj.unsqueeze(0)
                    # pool -> [Kj,D]
                    if pj.dim() == 4 and pj.size(1) == 1:
                        pj = pj.squeeze(1)     # [1,K,D] -> [K,D]
                    if pj.dim() == 3 and pj.size(0) == 1:
                        pj = pj.squeeze(0)     # [1,K,D] -> [K,D]
                    if pj.dim() == 2 and pj.size(0) == 1 and pj.size(1) == sj.size(-1):
                        # [1,D] -> 单动作，兼容
                        pass
                    Kj = pj.size(0) if pj.dim() == 2 else 1
                    if pj.dim() == 1:
                        pj = pj.unsqueeze(0)   # [D] -> [1,D]
                        Kj = 1
                    rep_sj = sj.expand(Kj, *sj.shape[1:])  # [Kj,M,D]
                    q_all_j = target_net(pj, rep_sj).squeeze(-1)  # [Kj]
                    q_all_j = torch.nan_to_num(q_all_j, nan=0.0, posinf=1e6, neginf=-1e6)
                    q_next_max_list.append(q_all_j.max())
                q_next_max = torch.stack(q_next_max_list, dim=0)  # [B_nf]
                next_state_values[non_final_mask] = q_next_max

    # ---- Bellman target & loss ----
    target = reward_batch + GAMMA * next_state_values
    target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)

    loss = F.smooth_l1_loss(q_sa, target)
    print("[DEBUG] loss:", float(loss.item()))
    if torch.isnan(loss) or torch.isinf(loss):
        print("[ERROR] Invalid loss detected!")
        print("q_sa:", q_sa)
        print("target:", target)
        return

    # ---- Backprop ----
    optimizerP.zero_grad(set_to_none=True)
    loss.backward()

    # 梯度裁剪 + 打印
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=grad_clip)
    with torch.no_grad():
        total_norm = 0.0
        print("\n[GRAD DEBUG] ====== Policy Net Gradients ======")
        for name, p in policy_net.named_parameters():
            if p.grad is None:
                print(f"[WARN] No grad in {name}")
                continue
            grad = torch.nan_to_num(p.grad, nan=0.0, posinf=1e6, neginf=-1e6)
            gnorm = grad.norm().item()
            total_norm += gnorm ** 2
            if "cross_attention" in name:
                tag = "[ATTN]"
            elif "ffn" in name:
                tag = "[FFN]"
            elif "q_predictor" in name:
                tag = "[HEAD]"
            else:
                tag = "[OTHER]"
            print(f"{tag:8s} {name:40s} | mean={grad.mean():.2e}, std={grad.std():.2e}, norm={gnorm:.2e}")
        total_norm = total_norm ** 0.5
        print(f"[GRAD DEBUG] Total grad norm: {total_norm:.4f}")
        print("[GRAD DEBUG] ====================================\n")

    optimizerP.step()

    optimize_model_conv.last_loss = float(loss.item())
    optimize_model_conv.last_q_mean = float(q_sa.mean().item())
    optimize_model_conv.last_reward_mean = float(reward_batch.mean().item())
    print("=== [DEBUG] Optimization done ===")


# def optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP,
#                         GAMMA, BATCH_SIZE):
#     print("\n=== [DEBUG] Enter optimize_model_conv ===")
#     print(f"len(memory): {len(memory)}, BATCH_SIZE: {BATCH_SIZE}")

#     if len(memory) < BATCH_SIZE:
#         print("[DEBUG] Memory not enough, skipping update.")
#         return

#     # ---- Sample ----
#     transitions = memory.sample(BATCH_SIZE)
#     batch = Transition(*zip(*transitions))

#     if random.random() < 0.02:  # 约每50次打印一次
#         print("[DEBUG] Sample example:")
#         print("  reward:", batch.reward[0].item() if torch.is_tensor(batch.reward[0]) else batch.reward[0])
#         print("  action:", batch.action[0])
#         print("  state_pool shape:", batch.state_pool[0].shape)
#         print("  next_state_pool is None?", batch.next_state_pool[0] is None)

#     # ---- Masks ----
#     non_final_mask = torch.tensor(
#         tuple(map(lambda s: s is not None, batch.next_state_pool)),
#         device='cuda', dtype=torch.bool
#     )
#     print("[DEBUG] non_final_mask shape:", non_final_mask.shape,
#           "sum:", non_final_mask.sum().item())

#     try:
#         non_final_next_states = torch.cat([s for s in batch.next_state_pool if s is not None])
#         non_final_next_state_subset = torch.cat([s for s in batch.next_state_subset if s is not None])
#         print("[DEBUG] non_final_next_states shape:", non_final_next_states.shape)
#     except Exception as e:
#         print("[ERROR] non_final_next_states concat failed:", e)
#         return

#     # ---- Batches ----
#     state_batch_pool = torch.cat(batch.state_pool)
#     state_batch_subset = torch.cat(batch.state_subset)
#     action_batch = torch.cat([a.unsqueeze(0) for a in batch.action])
#     reward_batch = torch.cat([r.unsqueeze(0) for r in batch.reward])

#     for i, a in enumerate(batch.action):
#         if a.dim() == 0:
#             print(f"[WARN] batch.action[{i}] is scalar:", a)


#     print("[DEBUG] state_batch_pool:", state_batch_pool.shape)
#     print("[DEBUG] state_batch_subset:", state_batch_subset.shape)
#     print("[DEBUG] action_batch:", action_batch.shape,
#           "min:", action_batch.min().item(), "max:", action_batch.max().item())
#     print("[DEBUG] reward_batch:", reward_batch.shape,
#           "min:", reward_batch.min().item(), "max:", reward_batch.max().item())

#     # ---- Forward ----
#     q_val = policy_net(state_batch_pool.cuda(), state_batch_subset.cuda())
#     print("[DEBUG] q_val shape:", q_val.shape)

#     # ---- Gather safety check ----
#     # if (action_batch >= q_val.size(1)).any():
#     #     print("[ERROR] action index out of bounds!",
#     #           "action_batch.max():", action_batch.max().item(),
#     #           "q_val.size(1):", q_val.size(1))
#     #     action_batch = action_batch.clamp(0, q_val.size(1) - 1)

#     state_action_values = q_val.squeeze(-1)  # 直接取预测的Q值
#     print("[DEBUG] state_action_values:", state_action_values.shape)

#     # ---- Next Q computation ----
#     next_state_values = torch.zeros(BATCH_SIZE, device='cuda')

#     if non_final_mask.sum().item() > 0:
#         target_out = target_net(non_final_next_states.cuda(), non_final_next_state_subset.cuda())
#         print("[DEBUG] target_out shape:", target_out.shape)
#         next_state_values[non_final_mask] = target_out.squeeze(-1).detach()

#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#     print("[DEBUG] expected_state_action_values:", expected_state_action_values.shape)

#     # ---- Loss ----
#     state_action_values = state_action_values.view(-1, 1)  # [B, 1]
#     expected_state_action_values = expected_state_action_values.view(-1, 1)  # [B, 1]
#     loss = F.smooth_l1_loss(state_action_values,
#                             expected_state_action_values.unsqueeze(1))
#     print("[DEBUG] loss:", loss.item())

#     # ---- Sanity checks ----
#     if torch.isnan(loss) or torch.isinf(loss):
#         print("[ERROR] Invalid loss detected!")
#         print("state_action_values:", state_action_values)
#         print("expected_state_action_values:", expected_state_action_values)
#         return


#     # ---- Backprop ----
#     optimizerP.zero_grad()
#     loss.backward()
#     for name, param in policy_net.named_parameters():
#         if param.grad is None:
#             print(f"[WARN] No grad in {name}")
#         else:
#             print(f"[DEBUG] grad[{name}]: mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")
#     optimizerP.step()

#     optimize_model_conv.last_loss = loss.item()
#     optimize_model_conv.last_q_mean = q_val.mean().item()
#     optimize_model_conv.last_reward_mean = reward_batch.mean().item()

#     print("=== [DEBUG] Optimization done ===")


# def optimize_model_conv(args, memory, Transition, policy_net, target_net, optimizerP, BATCH_SIZE=32, GAMMA=0.999,
#                         dqn_epochs=1):
#     """
#     此版本适配包含 state_pool, state_subset 等多个独立字段的 Transition 结构。
#     """
#     if len(memory) < BATCH_SIZE:
#         return

#     print('Optimizing policy network...')

#     policy_net.train()
#     loss_item = 0

#     for ep in range(dqn_epochs):
#         optimizerP.zero_grad()
#         transitions = memory.sample(BATCH_SIZE)
#         batch = Transition(*zip(*transitions))

#         # 1. 创建非终止状态的掩码
#         non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                                 batch.next_state_pool)),  # 使用 next_state_pool 检查
#                                       dtype=torch.bool, device='cuda')

#         # 2. 准备下一状态的两个部分
#         non_final_next_states_pool = torch.cat([s for s in batch.next_state_pool if s is not None])
#         non_final_next_states_subset = torch.cat([s for s in batch.next_state_subset if s is not None])

#         # 3. 准备当前状态的两个部分
#         state_batch_pool = torch.cat(batch.state_pool)
#         state_batch_subset = torch.cat(batch.state_subset)

#         # 4. 准备 action 和 reward
#         action_batch = torch.stack(batch.action).long().cuda()  # ✅ This is the correct fix
#         reward_batch = torch.stack(batch.reward).cuda()

#         # 计算 Q(s_t, a)
#         q_val = policy_net(state_batch_pool.cuda(), state_batch_subset.cuda())

#         # print("q_val.shape:", q_val.shape)
#         # print("action_batch.shape:", action_batch.shape)
#         # print("action_batch unique values:", action_batch.unique())

#         state_action_values = q_val.gather(1, action_batch.unsqueeze(1)).squeeze(1)
#         # state_action_values = q_val.squeeze(-1)  # 或 q_val.flatten()，得到 [B]

#         # 计算 V(s_{t+1})
#         next_state_values = torch.zeros(BATCH_SIZE, device='cuda')

#         print("=== optimize_model_conv debug ===")
#         print("batch size:", BATCH_SIZE)
#         print("non_final_mask:", non_final_mask)
#         print("non_final_mask.sum():", non_final_mask.sum())
#         print("actions shape:", action_batch.shape)
#         print("actions max:", action_batch.max().item(), "min:", action_batch.min().item())
#         print("q_val shape:", q_val.shape)

#         if non_final_mask.sum().item() > 0:
#             # 使用 Double DQN 逻辑
#             next_q_values_policy = policy_net(non_final_next_states_pool.cuda(),
#                                               non_final_next_states_subset.cuda()).detach()
#             best_actions = next_q_values_policy.max(1)[1].unsqueeze(1)
#             next_q_values_target = target_net(non_final_next_states_pool.cuda(),
#                                               non_final_next_states_subset.cuda()).detach()
#             next_state_values[non_final_mask] = next_q_values_target.gather(1, best_actions).squeeze()

#         # 计算期望的 Q 值
#         expected_state_action_values = (next_state_values * GAMMA) + reward_batch

#         # 计算 Huber loss
#         loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

#         loss_item += loss.item()
#         progress_bar(ep, dqn_epochs, '[DQN loss %.5f]' % (loss_item / (ep + 1)))
        
#         loss.backward()
#         total_grad = 0.0
#         for n, p in policy_net.named_parameters():
#             if p.grad is not None:
#                 total_grad += p.grad.norm().item()
#         print(f"[GradNorm] {total_grad:.6f}")

#         # REASON: 强化学习的训练过程容易不稳定，可能会产生非常大的梯度，
#         #         导致网络权重更新过猛（“梯度爆炸”），从而破坏学习过程。
#         #         梯度裁剪将所有参数的梯度范数强制限制在一个最大值（这里是1.0）以内，
#         #         确保了每次更新的步长都是合理的，从而极大地稳定了训练。
#         torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
#         optimizerP.step()

#         del (q_val)
#         del (expected_state_action_values)
#         del (loss)
#         del (next_state_values)
#         del (reward_batch)
#         # if non_final_mask.sum().item() > 0:
#         #     del (act)
#         #     del (v_val)
#         #     del (v_val_act)
#         del (state_action_values)
#         # del (state_batch)
#         del (action_batch)
#         del (non_final_mask)
#         # del (non_final_next_states)
#         del (batch)
#         del (transitions)
#     lab_set = open(os.path.join(args.ckpt_path, args.exp_name, 'q_loss.txt'), 'a')
#     lab_set.write("%f" % (loss_item))
#     lab_set.write("\n")
#     lab_set.close()

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
