# 文件名: wcf00317/alrl/alrl-reward_model/utils/al_scoring.py


import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# 👇 引入 feature_extractor
from utils.feature_extractor import UnifiedFeatureExtractor


def precompute_data_for_scoring(args, model, video_indices, train_set, batch_size=32, device="cuda"):
    """
    使用 UnifiedFeatureExtractor 统一提取 embedding，避免和模型接口耦合。
    """
    model.eval()
    print("正在为所有评分函数预计算所需数据...")

    # ✅ 实例化特征提取器
    ufe = UnifiedFeatureExtractor(args)

    all_embeddings, all_probs = [], []
    all_fast_embeds, all_slow_embeds = [], []
    all_view2_embeds = []

    # 1. 批量处理未标注数据
    with torch.no_grad():
        for i in tqdm(range(0, len(video_indices), batch_size), desc="预计算未标注数据"):
            batch_indices = video_indices[i:i + batch_size]

            # ✅ 统一用 feature_extractor 提取
            features, probs = ufe.get_embeddings_and_probs(batch_indices, model, train_set)

            all_embeddings.append(features.cpu())
            all_probs.append(probs.cpu())

            # ⚡ 快/慢速版本仍需逐个 clip 构造
            if getattr(args, 'use_temporal_consistency_feature', False):
                fast_clips, slow_clips = [], []
                for vid_idx in batch_indices:
                    fast, slow = train_set.get_video_multi_speed(vid_idx, fast_frames=16, slow_frames=8)
                    fast_clips.append(fast)
                    #slow_clips.append(slow.repeat_interleave(2, dim=2))
                    slow_upsampled = F.interpolate(slow, size=(16, slow.shape[3], slow.shape[4]), mode='trilinear',
                                                   align_corners=False)
                    slow_clips.append(slow_upsampled)

                fast_features, _ = ufe.get_embeddings_and_probs(batch_indices, model, train_set, video_tensors=fast_clips)
                slow_features, _ = ufe.get_embeddings_and_probs(batch_indices, model, train_set, video_tensors=slow_clips)

                all_fast_embeds.append(fast_features.cpu())
                all_slow_embeds.append(slow_features.cpu())

            if getattr(args, 'use_cross_view_consistency_feature', False):
                view2_clips = [train_set.get_video_augmented_views(vid_idx)[1] for vid_idx in batch_indices]
                view2_features, _ = ufe.get_embeddings_and_probs(batch_indices, model, train_set,
                                                                 video_tensors=view2_clips)
                all_view2_embeds.append(view2_features.cpu())

    # 2. 已标注数据
    labeled_indices = list(train_set.labeled_video_ids)
    labeled_embeddings = []
    if labeled_indices:
        with torch.no_grad():
            for i in tqdm(range(0, len(labeled_indices), batch_size), desc="预计算已标注嵌入"):
                batch_indices = labeled_indices[i:i + batch_size]

                # ✅ 统一用 feature_extractor
                features, _ = ufe.get_embeddings_and_probs(batch_indices, model, train_set)
                labeled_embeddings.append(features.cpu())

    return {
        'embeddings': torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0),
        'probs': torch.cat(all_probs, dim=0) if all_probs else torch.empty(0),
        'labeled_embeddings': torch.cat(labeled_embeddings, dim=0) if labeled_embeddings else None,
        'fast_embeds': torch.cat(all_fast_embeds, dim=0) if all_fast_embeds else torch.empty(0),
        'slow_embeds': torch.cat(all_slow_embeds, dim=0) if all_slow_embeds else torch.empty(0),
        'view2_embeds': torch.cat(all_view2_embeds, dim=0) if all_view2_embeds else torch.empty(0)
    }


# --- 以下是基于预计算数据的评分函数 ---

def compute_entropy_score(precomputed_data):
    """(1) 基于预计算的概率计算熵分数（对应 use_statistical_features）。"""
    probs = precomputed_data['probs']
    if probs.shape[0] == 0: return torch.empty(0)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

def compute_diversity_score(precomputed_data):
    """基于 batch 内 embeddings 的平均相似度计算多样性（相似度越高，多样性越低）。"""
    embeddings = precomputed_data['embeddings']
    # print("DEBUG:", embeddings.shape)

    if embeddings.ndim > 2:
        # 如果是 [N, C, T, H, W] 这类5D，取时空平均
        if embeddings.ndim == 5:
            embeddings = embeddings.mean(dim=(2, 3, 4))
        elif embeddings.ndim == 4:
            embeddings = embeddings.mean(dim=(2, 3))
        elif embeddings.ndim == 3:
            embeddings = embeddings.mean(dim=2)
        else:
            raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")

    if embeddings.shape[0] < 2:
        return torch.zeros(embeddings.shape[0])

    normed_embeds = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(normed_embeds, normed_embeds.T)  # [N, N] cosine similarity
    sim_matrix.fill_diagonal_(0.0)  # 去掉自相似
    mean_sim = sim_matrix.mean(dim=1)
    diversity_scores = 1.0 - mean_sim   # 越低表示越相似，越高表示更 diverse
    return diversity_scores

# def compute_diversity_score(precomputed_data):
#     embeddings = precomputed_data['embeddings']
#     print("DEBUG:", embeddings.shape)
#     if embeddings.shape[0] < 2:
#         return torch.zeros(embeddings.shape[0])

#     # --- 🔧 强制二维化 ---
#     if embeddings.ndim > 2:
#         embeddings = embeddings.reshape(-1, embeddings.shape[-1])

#     normed_embeds = F.normalize(embeddings, dim=1)
#     sim_matrix = torch.matmul(normed_embeds, normed_embeds.T)
#     sim_matrix.fill_diagonal_(0.0)
#     mean_sim = sim_matrix.mean(dim=1)
#     diversity_scores = 1.0 - mean_sim
#     return diversity_scores



# def compute_representativeness_score(precomputed_data):
#     """(3) 基于预计算的嵌入计算代表性分数（到质心的余弦相似度）。"""
#     embeddings = precomputed_data['embeddings']
#     if embeddings.shape[0] == 0: return torch.empty(0)
#     centroid = torch.mean(embeddings, dim=0, keepdim=True)
#     return F.cosine_similarity(embeddings, centroid.squeeze(0), dim=1)

def compute_representativeness_score(precomputed_data):
    """(3) 基于预计算的嵌入计算代表性分数（到质心的余弦相似度）。"""
    embeddings = precomputed_data['embeddings']
    if embeddings.shape[0] == 0:
        return torch.empty(0)

    # --- 🔧 统一维度: 将 [N, C, T, H, W] -> [N, C] ---
    if embeddings.ndim == 5:
        embeddings = embeddings.mean(dim=(2, 3, 4))
    elif embeddings.ndim == 4:
        embeddings = embeddings.mean(dim=(2, 3))
    elif embeddings.ndim == 3:
        embeddings = embeddings.mean(dim=2)
    elif embeddings.ndim != 2:
        raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")

    centroid = torch.mean(embeddings, dim=0, keepdim=True)
    return F.cosine_similarity(embeddings, centroid.squeeze(0), dim=1)


def compute_prediction_margin_score(precomputed_data):
    """(4) 基于预计算的概率计算预测边际分数。"""
    probs = precomputed_data['probs']
    if probs.shape[0] == 0: return torch.empty(0)
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    return 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])

# def compute_labeled_distance_score(precomputed_data):
#     """(5) 基于预计算的嵌入计算与已标注集的最小距离分数。"""
#     embeddings = precomputed_data['embeddings']
#     labeled_embeddings = precomputed_data['labeled_embeddings']
#     if labeled_embeddings is None or labeled_embeddings.shape[0] == 0:
#         return torch.zeros(embeddings.shape[0])
#     dist_matrix = torch.cdist(embeddings, labeled_embeddings)
#     scores, _ = torch.min(dist_matrix, dim=1)
#     return scores
def compute_labeled_distance_score(precomputed_data):
    """(5) 基于预计算的嵌入计算与已标注集的最小距离分数。"""
    embeddings = precomputed_data['embeddings']
    labeled_embeddings = precomputed_data['labeled_embeddings']

    # --- 🧩 统一维度: 如果是 [N, C, T, H, W] 或 [N, C, H, W] 等，取时空平均 ---
    def flatten_embeddings(x):
        if x is None:
            return None
        if x.ndim == 5:
            return x.mean(dim=(2, 3, 4))
        elif x.ndim == 4:
            return x.mean(dim=(2, 3))
        elif x.ndim == 3:
            return x.mean(dim=2)
        elif x.ndim == 2:
            return x
        else:
            raise ValueError(f"Unexpected embedding shape: {x.shape}")

    embeddings = flatten_embeddings(embeddings)
    labeled_embeddings = flatten_embeddings(labeled_embeddings)

    if labeled_embeddings is None or labeled_embeddings.shape[0] == 0:
        return torch.zeros(embeddings.shape[0])

    # ✅ 确认特征维度一致
    assert embeddings.shape[1] == labeled_embeddings.shape[1], \
        f"Embedding dim mismatch: unlabeled {embeddings.shape}, labeled {labeled_embeddings.shape}"

    dist_matrix = torch.cdist(embeddings, labeled_embeddings)  # [N_unlab, N_lab]
    scores, _ = torch.min(dist_matrix, dim=1)
    return scores

def compute_neighborhood_density_score(precomputed_data, k=10):
    embeddings = precomputed_data['embeddings']

    # --- 🔧 统一维度 ---
    if embeddings.ndim == 5:
        embeddings = embeddings.mean(dim=(2, 3, 4))
    elif embeddings.ndim == 4:
        embeddings = embeddings.mean(dim=(2, 3))
    elif embeddings.ndim == 3:
        embeddings = embeddings.mean(dim=2)
    elif embeddings.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

    n = embeddings.shape[0]
    if n <= k:
        return torch.zeros(n)

    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    dist_matrix.fill_diagonal_(float('inf'))

    knn_dists = torch.topk(dist_matrix, k, largest=False, dim=1).values
    mean_knn_dist = knn_dists.mean(dim=1)
    return 1.0 / (1.0 + mean_knn_dist)

def compute_temporal_consistency_score(precomputed_data):
    """(7) 基于预计算的快慢速嵌入计算时间一致性分数（每个视频一个分数）。"""
    fast_embeds = precomputed_data['fast_embeds']
    slow_embeds = precomputed_data['slow_embeds']

    if fast_embeds.shape[0] == 0:
        return torch.empty(0)

    # --- 🔧 统一形状 [N, C, T, H, W] -> [N, C]
    if fast_embeds.ndim == 5:
        fast_embeds = fast_embeds.mean(dim=(2, 3, 4))
        slow_embeds = slow_embeds.mean(dim=(2, 3, 4))
    elif fast_embeds.ndim == 4:
        fast_embeds = fast_embeds.mean(dim=(2, 3))
        slow_embeds = slow_embeds.mean(dim=(2, 3))
    elif fast_embeds.ndim == 3:
        fast_embeds = fast_embeds.mean(dim=2)
        slow_embeds = slow_embeds.mean(dim=2)
    elif fast_embeds.ndim != 2:
        raise ValueError(f"Unexpected embedding shape: {fast_embeds.shape}")

    # --- 🔁 计算每个视频的一致性分数
    return 1.0 - F.cosine_similarity(fast_embeds, slow_embeds, dim=1)

# def compute_neighborhood_density_score(precomputed_data, k=10):
#     """基于预计算的 embeddings 计算邻域密度分数。
#     分数定义为样本与其 k 近邻的平均距离的倒数。
#     """
#     embeddings = precomputed_data['embeddings']
#     n = embeddings.shape[0]
#     if n <= k:
#         return torch.zeros(n)

#     dist_matrix = torch.cdist(embeddings, embeddings, p=2)
#     dist_matrix.fill_diagonal_(float('inf'))  # 去掉自距离 0

#     knn_dists = torch.topk(dist_matrix, k, largest=False, dim=1).values  # [N, k]
#     mean_knn_dist = knn_dists.mean(dim=1)  # 每个样本的平均kNN距离
#     return 1.0 / (1.0 + mean_knn_dist)


# def compute_temporal_consistency_score(precomputed_data):
#     """(7) 基于预计算的快慢速嵌入计算时间一致性分数。"""
#     fast_embeds = precomputed_data['fast_embeds']
#     slow_embeds = precomputed_data['slow_embeds']
#     if fast_embeds.shape[0] == 0: return torch.empty(0)
#     return 1.0 - F.cosine_similarity(fast_embeds, slow_embeds, dim=1)


# --- 以下是需要独立计算的评分函数 ---

def compute_bald_score(model, video_indices, train_set, mc_dropout_iterations=10, batch_size=16):
    """使用 MC-Dropout 近似计算 BALD 分数。"""
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    print(f"正在使用 MC-Dropout (T={mc_dropout_iterations}) 计算 BALD 分数...")
    all_bald_scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(video_indices), batch_size), desc="计算 BALD 分数"):
            batch_indices = video_indices[i:i + batch_size]
            clips = torch.stack([train_set.get_video(idx) for idx in batch_indices], dim=0).cuda()

            batch_probs_mc = [F.softmax(model.cls_head(model.extract_feat(clips)[0]), dim=1).unsqueeze(0) for _ in
                              range(mc_dropout_iterations)]
            batch_probs_mc = torch.cat(batch_probs_mc, dim=0)

            mean_probs = torch.mean(batch_probs_mc, dim=0)
            entropy_of_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
            mean_of_entropies = torch.mean(-torch.sum(batch_probs_mc * torch.log(batch_probs_mc + 1e-8), dim=2), dim=0)

            bald_scores = entropy_of_mean - mean_of_entropies
            all_bald_scores.append(bald_scores.cpu())

    return torch.cat(all_bald_scores, dim=0) if all_bald_scores else torch.empty(0)

def compute_egl_score(model, video_indices, train_set, device="cuda"):
    """标准 EGL（完全定义版）：在 GPU 上运行"""
    model.eval()
    print("正在计算标准 EGL 分数（逐样本逐类别梯度）...")

    egl_scores = []
    last_layer = model.cls_head
    num_classes = last_layer.out_features

    for idx in tqdm(video_indices, desc="EGL per-sample"):
        clip = train_set.get_video(idx).unsqueeze(0).to(device)
        features = model.extract_feat(clip)[0]
        logits = last_layer(features)
        probs = F.softmax(logits, dim=1).detach().squeeze(0)  # [C]

        sample_score = torch.tensor(0.0, device=device)  # 保持在 GPU
        for y in range(num_classes):
            model.zero_grad()
            loss = F.cross_entropy(logits, torch.tensor([y], device=device), reduction='sum')
            loss.backward(retain_graph=True)

            params = [p for p in last_layer.parameters() if p.grad is not None]
            if params:
                all_grads = torch.cat([p.grad.flatten() for p in params])
                grad_norm = all_grads.norm(p=2)
                sample_score += probs[y] * grad_norm

        egl_scores.append(sample_score)

    # 一次性转回 CPU，避免频繁同步
    return torch.stack(egl_scores).detach().cpu()

def compute_egl_dynamic_score_optimized(model, video_indices, train_set, prob_threshold=0.95, batch_size=16):
    """
    动态 Top-K EGL (优化版): 每个样本只做一次 forward+backward
    """
    model.eval()
    all_scores = []
    last_layer = model.cls_head

    for i in tqdm(range(0, len(video_indices), batch_size), desc=f"计算 动态-EGL (优化) 分数"):
        batch_indices = video_indices[i:i + batch_size]
        clips = torch.stack([train_set.get_video(idx) for idx in batch_indices], dim=0).cuda()

        # 前向传播（一次）
        features = model.extract_feat(clips)[0]
        logits = last_layer(features)
        probs = F.softmax(logits, dim=1)

        batch_scores = []
        for j in range(len(batch_indices)):
            sample_probs = probs[j]
            sorted_probs, sorted_labels = torch.sort(sample_probs, descending=True)

            # 动态确定 K
            cumulative = torch.cumsum(sorted_probs, dim=0)
            k = int((cumulative < prob_threshold).sum().item() + 1)

            # 构造加权损失
            weighted_loss = 0.0
            for p_y, y in zip(sorted_probs[:k], sorted_labels[:k]):
                logit_j = logits[j:j+1]  # 单样本 logits
                loss_c = F.cross_entropy(logit_j, y.unsqueeze(0), reduction="sum")
                weighted_loss = weighted_loss + p_y * loss_c

            # 反向传播一次
            model.zero_grad()
            weighted_loss.backward(retain_graph=True)

            # 提取梯度范数
            params = [p for p in last_layer.parameters() if p.grad is not None]
            grads = torch.cat([p.grad.detach().flatten() for p in params])
            grad_norm = grads.norm(p=2).item()
            batch_scores.append(grad_norm)

        all_scores.extend(batch_scores)

    model.zero_grad()
    return torch.tensor(all_scores, dtype=torch.float32).cuda()


def compute_egl_adaptive_topk(model, video_indices, train_set, prob_threshold=0.95, batch_size=16):
    """
    自适应 Top-K EGL (优化版): 每个样本只做一次 forward+backward
    """
    model.eval()
    all_scores = []
    last_layer = model.cls_head

    all_k_values = []
    for i in tqdm(range(0, len(video_indices), batch_size), desc=f"计算自适应K-EGL分数"):
        batch_indices = video_indices[i:i + batch_size]
        # 假设 get_video 返回的是 [1, C, T, H, W]
        clips = torch.stack([train_set[idx][0] for idx in batch_indices], dim=0).cuda()

        # 前向传播（一次）
        features = model.extract_feat(clips)[0]
        logits = last_layer(features)
        probs = F.softmax(logits, dim=1)

        for j in range(len(batch_indices)):
            sample_probs = probs[j]
            sorted_probs, sorted_labels = torch.sort(sample_probs, descending=True)

            # 动态确定 K
            cumulative = torch.cumsum(sorted_probs, dim=0)
            k = int((cumulative < prob_threshold).sum().item() + 1)
            all_k_values.append(k)

            # 构造加权损失
            weighted_loss = 0.0
            for p_y, y in zip(sorted_probs[:k], sorted_labels[:k]):
                logit_j = logits[j:j + 1]  # 单样本 logits
                loss_c = F.cross_entropy(logit_j, y.unsqueeze(0), reduction="sum")
                weighted_loss = weighted_loss + p_y * loss_c

            # 反向传播一次
            model.zero_grad()
            # 最后一个样本无需保留图
            retain_graph = j < len(batch_indices) - 1
            weighted_loss.backward(retain_graph=retain_graph)

            # 提取梯度范数
            params = [p for p in last_layer.parameters() if p.grad is not None]
            grad_norm = 0.0
            if params:
                grads = torch.cat([p.grad.detach().flatten() for p in params])
                grad_norm = grads.norm(p=2).item()

            all_scores.append(grad_norm)
    if all_k_values:
        print(f"  - 动态 k 值统计: 平均值={np.mean(all_k_values):.2f}, 最小值={min(all_k_values)}, 最大值={max(all_k_values)}")
    model.zero_grad()
    return torch.tensor(all_scores, dtype=torch.float32)

def compute_egl_score_approx(model, video_indices, train_set, batch_size=16):
    """GEMINI版本的工程近似，会快很多！计算最后一层分类头的期望梯度长度 (EGL) 分数。"""
    model.eval()
    print("正在计算期望梯度长度 (EGL) 分数...")
    all_egl_scores = []
    last_layer = model.cls_head

    for i in tqdm(range(0, len(video_indices), batch_size), desc="计算 EGL 分数"):
        batch_indices = video_indices[i:i + batch_size]
        clips = torch.stack([train_set.get_video(idx) for idx in batch_indices], dim=0).cuda()

        features = model.extract_feat(clips)[0]
        # if features.dim() > 2: features = features.mean(dim=[2, 3, 4])
        logits = last_layer(features)
        pseudo_labels = torch.argmax(logits, dim=1).detach()
        loss = F.cross_entropy(logits, pseudo_labels, reduction='sum')

        model.zero_grad()
        loss.backward()

        params = [p for p in last_layer.parameters() if p.grad is not None]
        if not params:
            grad_norm = 0
        else:
            # 2. 将所有参数的梯度展平并拼接成一个长向量
            all_grads = torch.cat([p.grad.detach().flatten() for p in params])
            # 3. 计算这个长向量的L2范数
            grad_norm = all_grads.norm(p=2).item()

        avg_grad_norm = grad_norm / len(batch_indices)
        all_egl_scores.extend([avg_grad_norm] * len(batch_indices))

    model.zero_grad()
    return torch.tensor(all_egl_scores, dtype=torch.float32)
def compute_cross_view_consistency_score(precomputed_data):
    view1_embeds = precomputed_data['embeddings']
    view2_embeds = precomputed_data['view2_embeds']
    if view1_embeds.shape[0] == 0 or view2_embeds.shape[0] == 0:
        return torch.empty(0)
    return 1.0 - F.cosine_similarity(view1_embeds, view2_embeds, dim=1)