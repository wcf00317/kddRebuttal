# 文件名: wcf00317/alrl/alrl-reward_model/utils/feature_extractor.py

import torch
import torch.nn as nn  # --- NEW --- (为了使用 nn.Module)
import torch.nn.functional as F
from tqdm import tqdm


class UnifiedFeatureExtractor:
    def __init__(self, args):
        self.args = args
        self.active_features = []
        self.feature_dim = 0
        self.feature_dim_names = []  # <--- 新增此行: 初始化一个空列表

        print("\n--- Initializing Unified Feature Extractor ---")

        # --- 在这里，您可以像开关一样控制所有特征 ---
        # --- 在您的 .yaml 配置文件中设置这些参数为 true 或 false ---

        # 建议的Baseline特征
        if getattr(args, 'use_statistical_features', True):
            self.active_features.append('statistical')
            self.feature_dim += 4
            # <--- 新增下面这行: 添加统计特征的名称 ---
            self.feature_dim_names.extend(['mean_entropy', 'std_entropy', 'mean_similarity', 'std_similarity'])
            print("  - Statistical Features (熵/相似度统计): ENABLED")

        # 批内多样性
        if getattr(args, 'use_diversity_feature', False):
            self.active_features.append('diversity')
            self.feature_dim += 1
            # <--- 新增下面这行: 添加多样性特征的名称 ---
            self.feature_dim_names.append('diversity_score')
            print("  - Intra-Batch Diversity Feature: ENABLED")

        # 代表性
        if getattr(args, 'use_representativeness_feature', False):
            self.active_features.append('representativeness')
            self.feature_dim += 1
            # <--- 新增下面这行: 添加代表性特征的名称 ---
            self.feature_dim_names.append('representativeness_score')
            print("  - Representativeness Feature: ENABLED")

        # 邻域密度
        if getattr(args, 'use_neighborhood_density_feature', False):
            self.active_features.append('neighborhood_density')
            self.feature_dim += 1
            # <--- 新增下面这行: 添加邻域密度特征的名称 ---
            self.feature_dim_names.append('neighborhood_density_score')
            print("  - Neighborhood Density Feature: ENABLED")

        # 预测边际
        if getattr(args, 'use_prediction_margin_feature', False):
            self.active_features.append('prediction_margin')
            self.feature_dim += 2  # 均值和标准差
            # <--- 新增下面这行: 添加预测边际特征的名称 ---
            self.feature_dim_names.extend(['mean_inv_margin', 'std_inv_margin']) # 注意这里用 1-margin, 所以叫 inv_margin
            print("  - Prediction Margin Feature: ENABLED")

        # 与已标注集的距离
        if getattr(args, 'use_labeled_distance_feature', False):
            self.active_features.append('labeled_distance')
            self.feature_dim += 2  # 均值和标准差
            # <--- 新增下面这行: 添加距离特征的名称 ---
            self.feature_dim_names.extend(['mean_dist_to_labeled', 'std_dist_to_labeled'])
            print("  - Distance to Labeled Set Feature: ENABLED")

        # 时间一致性
        if getattr(args, 'use_temporal_consistency_feature', False):
            self.active_features.append('temporal_consistency')
            self.feature_dim += 2 # 均值和标准差
            # <--- 新增下面这行: 添加时间一致性特征的名称 ---
            self.feature_dim_names.extend(['mean_temporal_inconsistency', 'std_temporal_inconsistency'])
            print("  - Temporal Consistency Feature: ENABLED")


        if not self.active_features:
            raise ValueError("错误：至少需要启用一种特征！请检查您的配置文件。")

        print(f"Total feature dimension: {self.feature_dim}")
        print(f"Feature names: {self.feature_dim_names}") # <-- 可以加一行打印确认
        print("------------------------------------------\n")

    def extract_emb(self, batch_video_indices, model, train_set, all_unlabeled_embeddings=None, all_labeled_embeddings=None, precomputed_data=None):
        if not batch_video_indices:
            return torch.zeros(self.feature_dim)

        batch_embeddings, batch_probs = self.get_embeddings_and_probs(batch_video_indices, model, train_set)
        feature_tensors = []

        if 'statistical' in self.active_features:
            # --- 熵：用概率分布计算 per-sample entropy，然后做批内聚合 ---
            probs = torch.clamp(batch_probs, min=1e-8, max=1.0)  # 数值安全
            entropies = -(probs * probs.log()).sum(dim=1)  # [B]
            mean_entropy = entropies.mean().item()
            std_entropy = entropies.std().item() if entropies.numel() > 1 else 0.0

            # --- 相似度：批内两两余弦相似度的上三角均值/方差 ---
            emb2d = batch_embeddings
            if emb2d.dim() > 2:  # 聚合到 [B, C]
                emb2d = emb2d.mean(dim=tuple(range(2, emb2d.dim())))
            if emb2d.size(0) > 1:
                normed = F.normalize(emb2d, p=2, dim=1)
                cos = normed @ normed.t()  # [B, B]
                iu = torch.triu_indices(emb2d.size(0), emb2d.size(0), offset=1)
                sims = cos[iu[0], iu[1]]  # 上三角（不含对角）
                mean_similarity = sims.mean().item()
                std_similarity = sims.std().item() if sims.numel() > 1 else 0.0
            else:
                mean_similarity, std_similarity = 0.0, 0.0

            # 维度仍然是 4： [mean_entropy, std_entropy, mean_similarity, std_similarity]
            feature_tensors.append(torch.tensor([mean_entropy, std_entropy, mean_similarity, std_similarity]))

        if 'diversity' in self.active_features:
            diversity_score = 0.0
            if len(batch_video_indices) > 1:
                normed_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                cosine_sim_matrix = torch.matmul(normed_embeddings, normed_embeddings.t())
                upper_tri_indices = torch.triu_indices(len(batch_video_indices), len(batch_video_indices), offset=1)
                mean_cosine_sim = cosine_sim_matrix[upper_tri_indices[0], upper_tri_indices[1]].mean().item()
                diversity_score = 1.0 - mean_cosine_sim
            feature_tensors.append(torch.tensor([diversity_score]))

        if 'representativeness' in self.active_features:
            representativeness_score = 0.0
            if all_unlabeled_embeddings is not None and len(all_unlabeled_embeddings) > 0:
                unlabeled_centroid = all_unlabeled_embeddings.mean(dim=0, keepdim=True).to(batch_embeddings.device)
                batch_centroid = batch_embeddings.mean(dim=0, keepdim=True)
                representativeness_score = F.cosine_similarity(unlabeled_centroid, batch_centroid).item()
            feature_tensors.append(torch.tensor([representativeness_score]))

        if 'neighborhood_density' in self.active_features:
            density_score = 0.0
            if all_unlabeled_embeddings is not None and len(all_unlabeled_embeddings) > 1:
                #dist_matrix = torch.cdist(batch_embeddings.cpu(), all_unlabeled_embeddings.cpu())
                dist_matrix = torch.cdist(batch_embeddings.to(device), all_unlabeled_embeddings.to(device))
                k = min(10, len(all_unlabeled_embeddings))
                knn_dists = torch.topk(dist_matrix, k, largest=False, dim=1).values
                mean_knn_dist = knn_dists.mean().item()
                density_score = 1.0 / (1.0 + mean_knn_dist)
            feature_tensors.append(torch.tensor([density_score]))

            # --- NEW: 增加新特征的计算逻辑 ---
        if 'prediction_margin' in self.active_features:
            # 对每个样本的概率分布进行排序
            sorted_probs, _ = torch.sort(batch_probs, dim=1, descending=True)
            # 边际 = (第一大概率) - (第二大概率)
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            mean_margin = margins.mean().item()
            std_margin = margins.std().item() if len(margins) > 1 else 0.0
            # 我们希望边际越小越好（模型越纠结），所以用 1.0 - margin 来表示不确定性的大小
            feature_tensors.append(torch.tensor([1.0 - mean_margin, std_margin]))

        if 'labeled_distance' in self.active_features:
            mean_dist = 0.0
            std_dist = 0.0
            if all_labeled_embeddings is not None and len(all_labeled_embeddings) > 0:
                # 计算批内每个样本到已标注集的最短距离
                #dist_matrix = torch.cdist(batch_embeddings.cpu(), all_labeled_embeddings.cpu())
                dist_matrix = torch.cdist(batch_embeddings.to(device), all_labeled_embeddings.to(device))
                min_dists, _ = torch.min(dist_matrix, dim=1)
                mean_dist = min_dists.mean().item()
                std_dist = min_dists.std().item() if len(min_dists) > 1 else 0.0
            feature_tensors.append(torch.tensor([mean_dist, std_dist]))
        # --- 新增特征的计算 ---
        if 'temporal_consistency' in self.active_features:
            batch_fast_embeds = []
            batch_slow_embeds = []
            with torch.no_grad():
                for vid_idx in batch_video_indices:
                    # 1. 独立获取快慢速视频
                    fast_clip, slow_clip = train_set.get_video_multi_speed(vid_idx, fast_frames=16, slow_frames=8)

                    # 2. 时间上采样以适配C3D
                    if slow_clip.shape[2] == 8:
                        slow_clip = slow_clip.repeat_interleave(2, dim=2)

                    # 3. 独立提取特征
                    fast_embed = model.extract_feat(fast_clip.unsqueeze(0).cuda())[0].cpu()
                    slow_embed = model.extract_feat(slow_clip.unsqueeze(0).cuda())[0].cpu()

                    batch_fast_embeds.append(fast_embed)
                    batch_slow_embeds.append(slow_embed)

            batch_fast_embeds = torch.cat(batch_fast_embeds, dim=0)
            batch_slow_embeds = torch.cat(batch_slow_embeds, dim=0)

            # 4. 计算并拼接特征
            inconsistency_scores = 1.0 - F.cosine_similarity(batch_fast_embeds, batch_slow_embeds, dim=1)
            mean_inconsistency = inconsistency_scores.mean().item()
            std_inconsistency = inconsistency_scores.std().item() if len(inconsistency_scores) > 1 else 0.0
            feature_tensors.append(torch.tensor([mean_inconsistency, std_inconsistency]))

        if 'cross_view_consistency' in self.active_features:
            if precomputed_data is None or 'view2_embeds' not in precomputed_data:
                raise ValueError("Cross-view consistency需要预计算的'view2_embeds'，但未提供。")

            # 找到当前批次在整个未标注池中的位置索引
            unlabeled_indices = train_set.get_candidates_video_ids()
            pos_indices = [unlabeled_indices.index(vid) for vid in batch_video_indices]

            batch_view1_embeds = precomputed_data['embeddings'][pos_indices]
            batch_view2_embeds = precomputed_data['view2_embeds'][pos_indices]

            inconsistency_scores = 1.0 - F.cosine_similarity(batch_view1_embeds, batch_view2_embeds, dim=1)
            mean_inconsistency = inconsistency_scores.mean().item()
            std_inconsistency = inconsistency_scores.std().item() if len(inconsistency_scores) > 1 else 0.0
            feature_tensors.append(torch.tensor([mean_inconsistency, std_inconsistency]))
        return torch.cat(feature_tensors)

    def extract(self, batch_video_indices, model, train_set, batch_scores=None):
        """
        为一个批次的视频提取最终的特征向量。
        它现在接收一个包含该批次所有预计算分数的字典。
        """
        if not batch_video_indices:
            return torch.zeros(self.feature_dim)

        feature_tensors = []

        # --- 提取当前批次的基础信息 ---
        batch_embeddings, batch_probs = self.get_embeddings_and_probs(batch_video_indices, model, train_set)

        if 'statistical' in self.active_features:
            # 1) 批内熵的均值/方差（数值更稳）
            probs = torch.clamp(batch_probs, min=1e-8, max=1.0)
            entropies = -(probs * probs.log()).sum(dim=1)  # [B]
            mean_entropy = entropies.mean().item()
            std_entropy = entropies.std().item() if entropies.numel() > 1 else 0.0

            # 2) 批内两两余弦相似度的均值/方差（替代原来的 0.5 占位）
            emb2d = batch_embeddings
            if emb2d.dim() > 2:  # 聚合到 [B, C]
                emb2d = emb2d.mean(dim=tuple(range(2, emb2d.dim())))
            B = emb2d.size(0)
            if B > 1:
                normed = F.normalize(emb2d, p=2, dim=1, eps=1e-8)
                cos = normed @ normed.t()  # [B, B]
                iu = torch.triu_indices(B, B, offset=1)  # 上三角(不含对角)
                sims = cos[iu[0], iu[1]]
                mean_similarity = sims.mean().item()
                std_similarity = sims.std().item() if sims.numel() > 1 else 0.0
            else:
                mean_similarity, std_similarity = 0.0, 0.0

            feature_tensors.append(torch.tensor([
                mean_entropy, std_entropy, mean_similarity, std_similarity
            ]))
        # 所有其他策略都直接从 batch_scores 中获取结果
        if 'diversity' in self.active_features:
            # 多样性是批次内计算的，不依赖外部 precomputed_data
            diversity_score = 0.0
            if batch_embeddings.ndim == 5:
                batch_embeddings = batch_embeddings.mean(dim=(2, 3, 4))  # [B, C]
            elif batch_embeddings.ndim == 4:
                batch_embeddings = batch_embeddings.mean(dim=(2, 3))      # [B, C]
            elif batch_embeddings.ndim == 3:
                batch_embeddings = batch_embeddings.mean(dim=2)           # [B, C]
            elif batch_embeddings.ndim != 2:
                raise ValueError(f"Unexpected embedding shape: {batch_embeddings.shape}")

            if len(batch_video_indices) > 1:
                normed_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                cosine_sim_matrix = torch.matmul(normed_embeddings, normed_embeddings.t())
                upper_tri_indices = torch.triu_indices(len(batch_video_indices), len(batch_video_indices), offset=1)
                diversity_score = 1.0 - cosine_sim_matrix[upper_tri_indices[0], upper_tri_indices[1]].mean().item()
            feature_tensors.append(torch.tensor([diversity_score]))

        # 对于需要预计算的特征，我们直接从传入的scores字典聚合
        aggregation_map = {
            'representativeness': 'representativeness',
            'prediction_margin': 'prediction_margin',
            'labeled_distance': 'labeled_distance',
            'neighborhood_density': 'neighborhood_density',
            'temporal_consistency': 'temporal_consistency',
        }

        for feature_name, score_key in aggregation_map.items():
            if feature_name in self.active_features:
                if batch_scores is None or score_key not in batch_scores:
                    # 如果启用的特征在传入的分数中找不到，就打印一个警告并跳过，而不是直接报错
                    print(f"警告：特征 '{feature_name}' 已启用，但在 batch_scores 中未提供其分数，将跳过此特征的提取。")
                    continue

                scores_tensor = batch_scores[score_key]
                mean_score = scores_tensor.mean().item()
                std_score = scores_tensor.std().item() if len(scores_tensor) > 1 else 0.0

                # 特殊处理：margin分数需要转换
                if feature_name == 'prediction_margin':
                    feature_tensors.append(torch.tensor([1.0 - mean_score, std_score]))
                elif feature_name in ['representativeness', 'neighborhood_density']:
                    # 代表性和邻域密度: 只添加 mean (1维)
                    feature_tensors.append(torch.tensor([mean_score]))
                else:
                    # 其他成对特征 (labeled_distance, temporal_consistency, cross_view): 添加 mean 和 std (2维)
                    feature_tensors.append(torch.tensor([mean_score, std_score]))

        final_features = torch.cat(feature_tensors)
        expected_dim = self.feature_dim # 这里的 feature_dim 应该是 13
        if final_features.numel() != expected_dim:
            # (这里的警告现在不应该再触发了)
            print(f"警告: extract 方法输出维度 ({final_features.numel()}) 与 __init__ 计算维度 ({expected_dim}) 不符!")
        return final_features

    def get_embeddings_and_probs(self, video_indices, model, train_set, video_tensors=None):
        model.eval()
        batch_embeddings = []
        batch_probs = []

        with torch.no_grad():
            # 如果没有提供预加载的tensors，则像原来一样从train_set加载
            if video_tensors is None:
                video_tensors = [train_set.get_video(vid_idx) for vid_idx in video_indices]

            # 统一处理张量列表
            for video_tensor in video_tensors:
                # 确保输入有batch维度并移动到GPU
                if video_tensor.dim() == 5:
                    video_tensor = video_tensor.unsqueeze(0)
                video_tensor = video_tensor.cuda()
                # print(video_tensor.shape)
                features = model.extract_feat(video_tensor)[0]
                # if features.ndim == 5:
                #     features = torch.mean(features, dim=[2, 3, 4])  # [B, C]
                # elif features.ndim == 4:
                #     features = torch.mean(features, dim=[2, 3])     # [B, C]
                # elif features.ndim == 3:
                #     features = torch.mean(features, dim=2)          # [B, C]
                # # 最终都应该是 [B, C]
                if features.shape[0] > 1:
                    features = features.mean(dim=0, keepdim=True)
                # if features.shape[0] > 1: features = features.mean(dim=0, keepdim=True)
                batch_embeddings.append(features)

                logits = model.cls_head(features)
                probs = F.softmax(logits, dim=1)
                batch_probs.append(probs)

        return torch.cat(batch_embeddings, dim=0), torch.cat(batch_probs, dim=0)

def get_all_unlabeled_embeddings(args, model, train_set):
    # ... (此函数保持不变) ...
    unlabeled_indices = train_set.get_candidates_video_ids()
    print(f"正在为 {len(unlabeled_indices)} 个未标注视频预计算特征嵌入...")
    all_embeddings = []
    batch_size = args.val_batch_size
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(unlabeled_indices), batch_size), desc="Pre-computing embeddings"):
            batch_indices = unlabeled_indices[i:i + batch_size]
            videos = [train_set.get_video(idx) for idx in batch_indices]
            video_batch_tensor = torch.cat(videos, dim=0).cuda()
            video_batch_tensor = video_batch_tensor.unsqueeze(0).cuda()
            features = model.extract_feat(video_batch_tensor)[0]
            if features.dim() > 2: features = features.mean(dim=0)
            all_embeddings.append(features.cpu())
    return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, args.embed_dim)
    # return torch.cat(all_embeddings, dim=0) if all_embeddings else None

# --- NEW: 增加一个辅助函数来获取已标注样本的嵌入 ---
def get_all_labeled_embeddings(args, model, train_set):
    """
    一个新增的辅助函数，用于计算所有已标注视频的特征嵌入。
    """
    labeled_indices = list(train_set.labeled_video_ids)
    if not labeled_indices:
        return None

    print(f"正在为 {len(labeled_indices)} 个已标注视频预计算特征嵌入...")

    all_embeddings = []
    batch_size = args.val_batch_size

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(labeled_indices), batch_size), desc="Pre-computing labeled embeddings"):
            batch_indices = labeled_indices[i:i + batch_size]
            # 假设 get_video 可以接受 is_eval 参数来使用评估时的数据增强
            videos = [train_set.get_video(idx) for idx in batch_indices]
            video_batch_tensor = torch.cat(videos, dim=0).cuda()
            video_batch_tensor = video_batch_tensor.unsqueeze(0).cuda()

            features = model.extract_feat(video_batch_tensor)[0]
            if features.dim() > 2: features = features.mean(dim=0)

            all_embeddings.append(features.cpu())

    return torch.cat(all_embeddings, dim=0) if all_embeddings else None
