# 文件名: train_ebm.py

import os
import pickle
import torch
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier


def train_ebm_reward_model(preference_data, exp_dir, feature_names=None):
    """
    使用差分特征训练 EBM 奖励模型（最终优化版）
    支持：
    - 自适应噪声增强（按特征方差）
    - 正负样本对称扰动
    - 特征标准化
    - 奖励诊断输出
    - 特征重要性报告保存
    """

    import numpy as np
    import torch
    import os
    import pickle
    from interpret.glassbox import ExplainableBoostingClassifier
    from sklearn.metrics import roc_auc_score

    print("\n--- 正在准备EBM训练数据 ---")
    if not preference_data:
        print("⚠️ 警告: 偏好数据为空，无法训练EBM模型。")
        return False

    # 1️⃣ 提取所有特征并计算全局均值
    all_features = []
    for pair in preference_data:
        all_features.append(pair['winner'])
        all_features.append(pair['loser'])
    all_features_tensor = torch.stack(all_features)
    global_mean_feature = torch.mean(all_features_tensor, dim=0)
    print(f"从 {len(all_features)} 个特征向量中计算出全局均值特征。")

    # 2️⃣ 预先计算每维标准差（用于自适应噪声）
    X_all = np.stack([(pair['winner'] - pair['loser']).numpy() for pair in preference_data])
    feat_std = X_all.std(axis=0, keepdims=True)
    noise_scale = 0.18  # 自适应噪声强度 (建议 0.05~0.1)
    print(f"[INFO] 使用自适应噪声增强，比例系数 noise_scale={noise_scale}")

    # 3️⃣ 构建训练数据
    diff_features, labels = [], []
    for pair in preference_data:
        diff_pos = pair['winner'] - pair['loser']
        diff_neg = pair['loser'] - pair['winner']

        # 自适应对称噪声
        noise = np.random.normal(0, 1.0, diff_pos.shape) * feat_std * noise_scale
        diff_pos_aug = diff_pos.numpy() + noise
        diff_neg_aug = diff_neg.numpy() - noise

        # 原始 + 增强数据（保证正负平衡）
        for x, y in [
            (diff_pos.numpy(), 1),
            (diff_neg.numpy(), 0),
            (diff_pos_aug, 1),
            (diff_neg_aug, 0)
        ]:
            diff_features.append(x.reshape(1, -1))
            labels.append(y)

        print(f"[DEBUG] diff_pos mean={diff_pos.mean():.4f}, std={diff_pos.std():.4f}")
        print(f"[DEBUG] diff_neg mean={diff_neg.mean():.4f}, std={diff_neg.std():.4f}")
        print(f"[DEBUG] noise mean={noise.mean():.4f}, std={noise.std():.4f}, ratio={noise.std()/ (diff_pos.std()+1e-6):.3f}")

    X_train = np.array(diff_features)
    y_train = np.array(labels)
    print(f"DEBUG: Shape of X_train before EBM fit: {X_train.shape}")

    # 4️⃣ 标准化
    X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-6)

    # 5️⃣ 训练 EBM 模型
    print("--- 开始训练EBM分类器 ---")
    ebm = ExplainableBoostingClassifier(
        random_state=42,
        interactions=2,     # 启用交互项
        max_bins=32,
        inner_bags=8,
        outer_bags=16,       # 增强稳定性
        learning_rate=0.05   # 稳定学习率
    )
    ebm.fit(X_train, y_train)
    print("--- EBM模型训练完成 ---")

    # 6️⃣ 性能诊断
    y_pred = ebm.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, y_pred)
    print(f"[DEBUG] EBM train AUC = {auc:.4f}")
    print(f"[DEBUG] Reward range: min={y_pred.min():.3f}, max={y_pred.max():.3f}, std={y_pred.std():.3f}")
    print(f"[DEBUG] Mean pos={y_pred[y_train==1].mean():.3f}, neg={y_pred[y_train==0].mean():.3f}")
    if y_pred.std() < 0.01:
        print("⚠️ 警告: 奖励模型输出方差极低，模型可能未学到有效区分特征。")

    # 7️⃣ 提取特征重要性
    print("--- 正在提取EBM特征重要性 ---")
    try:
        ebm_global = ebm.explain_global()
        explanation_data = ebm_global.data()
        output_path = os.path.join(exp_dir, 'ebm_feature_importance.txt')

        with open(output_path, 'w') as f:
            f.write("EBM Global Feature Importance\n")
            f.write("=============================\n")

            if feature_names and len(feature_names) == len(explanation_data['scores']):
                final_feature_names = feature_names
                print("使用传入的特征名称生成报告。")
            else:
                print("警告: 特征名长度不匹配，将使用默认索引名。")
                final_feature_names = [f"feature_{i}" for i in range(len(explanation_data['scores']))]

            for name, score in sorted(zip(final_feature_names, explanation_data['scores']),
                                      key=lambda x: x[1], reverse=True):
                f.write(f"{name}: {score:.6f}\n")

        print(f"✅ 特征重要性报告已保存至: {output_path}")

    except Exception as e:
        print(f"⚠️ 提取特征重要性时出错: {e}")

    # 8️⃣ 保存模型
    ebm_scorer = {'model': ebm, 'mean_feature': global_mean_feature}
    scorer_path = os.path.join(exp_dir, 'ebm_scorer.pkl')
    with open(scorer_path, 'wb') as f:
        pickle.dump(ebm_scorer, f)
    print(f"✅ EBM计分器已保存至: {scorer_path}")

    return True

# def train_ebm_reward_model(preference_data, exp_dir, feature_names=None):
#     """
#     使用拼接输入 [winner, loser] 的方式训练 EBM 奖励模型。
#     输出：ExplainableBoostingClassifier + 全局均值特征（mean_feature）
#     """
#     print("\n=== [EBM Trainer] 准备拼接式偏好数据 ===")

#     if not preference_data:
#         print("[ERROR] 偏好数据为空，无法训练奖励模型。")
#         return False

#     # 1️⃣ 汇总所有特征用于计算 mean_feature
#     all_features = []
#     for pair in preference_data:
#         all_features.append(pair['winner'])
#         all_features.append(pair['loser'])
#     all_features_tensor = torch.stack(all_features)
#     global_mean_feature = all_features_tensor.mean(dim=0).cpu()
#     print(f"[INFO] 从 {len(all_features)} 个样本计算 global_mean_feature，维度 {tuple(global_mean_feature.shape)}")

#     # 2️⃣ 构造拼接输入 + 标签
#     X_list, y_list = [], []
#     for pair in preference_data:
#         winner, loser = pair['winner'], pair['loser']
#         x_pos = torch.cat([winner, loser], dim=-1)  # label=1
#         x_neg = torch.cat([loser, winner], dim=-1)  # label=0
#         X_list.extend([x_pos.numpy(), x_neg.numpy()])
#         y_list.extend([1, 0])

#     X_train = np.stack(X_list, axis=0)
#     y_train = np.array(y_list)
#     print(f"[INFO] EBM训练输入形状: {X_train.shape}, 标签数量: {len(y_train)} (正样本={y_train.sum()})")

#     # 3️⃣ 初始化并训练 EBM 分类器
#     print("[EBM] 开始训练 ExplainableBoostingClassifier ...")
#     ebm = ExplainableBoostingClassifier(
#         random_state=42,
#         interactions=0,       # 禁用交互项，防止过拟合
#         max_bins=256,
#         outer_bags=4
#     )
#     ebm.fit(X_train, y_train)
#     print("[EBM] 模型训练完成 ✅")

#     # 4️⃣ 验证模型区分能力
#     try:
#         y_pred = ebm.predict_proba(X_train)[:, 1]
#         auc = roc_auc_score(y_train, y_pred)
#         print(f"[DEBUG] EBM train AUC={auc:.4f}, pred_mean={y_pred.mean():.4f}, pred_std={y_pred.std():.4f}")
#         print(f"[DEBUG] pred_min={y_pred.min():.4f}, pred_max={y_pred.max():.4f}")
#     except Exception as e:
#         print(f"[WARN] 无法计算训练AUC: {e}")

#     # 5️⃣ 提取特征重要性并保存报告
#     try:
#         ebm_global = ebm.explain_global()
#         data = ebm_global.data()
#         scores = data['scores']
#         feature_names_final = (
#             feature_names if feature_names is not None and len(feature_names) == len(scores)
#             else [f"feature_{i}" for i in range(len(scores))]
#         )

#         fi_path = os.path.join(exp_dir, 'ebm_feature_importance.txt')
#         with open(fi_path, 'w') as f:
#             f.write("EBM Global Feature Importance\n=============================\n")
#             for name, score in sorted(zip(feature_names_final, scores), key=lambda x: x[1], reverse=True):
#                 f.write(f"{name}: {score:.6f}\n")
#         print(f"[INFO] 特征重要性已保存至: {fi_path}")
#     except Exception as e:
#         print(f"[WARN] 无法保存特征重要性报告: {e}")

#     # 6️⃣ 保存模型 + 均值特征
#     ebm_scorer = {'model': ebm, 'mean_feature': global_mean_feature}
#     scorer_path = os.path.join(exp_dir, 'ebm_scorer.pkl')
#     with open(scorer_path, 'wb') as f:
#         pickle.dump(ebm_scorer, f)
#     print(f"[SAVE] 奖励模型已保存至: {scorer_path}")

#     return True

def load_ebm_scorer(exp_dir):
    """加载EBM计分器 (模型 + 均值特征)。"""
    scorer_path = os.path.join(exp_dir, 'ebm_scorer.pkl')
    if not os.path.exists(scorer_path):
        raise FileNotFoundError(f"找不到EBM计分器文件: {scorer_path}")
    with open(scorer_path, 'rb') as f:
        ebm_scorer = pickle.load(f)
    print("EBM计分器加载成功。")
    return ebm_scorer

# def predict_ebm_reward(ebm_scorer, batch_features):
#     """
#     支持输入 [D] 或 [N, D] 的 EBM 奖励预测函数。
#     """
#     ebm_model = ebm_scorer['model']
#     mean_feature = ebm_scorer['mean_feature']

#     if batch_features.device != mean_feature.device:
#         mean_feature = mean_feature.to(batch_features.device)

#     # 兼容单样本输入
#     if batch_features.dim() == 1:
#         batch_features = batch_features.unsqueeze(0)

#     N, D = batch_features.shape
#     mean_expand = mean_feature.unsqueeze(0).expand(N, -1)

#     x_input = torch.cat([batch_features, mean_expand], dim=-1)  # [N, 2D]
#     diff_np = x_input.cpu().numpy()

#     proba = ebm_model.predict_proba(diff_np)[:, 1]
#     rewards = torch.from_numpy(proba).float()

#     print(f"[EBM Reward] mean={rewards.mean():.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}, std={rewards.std():.4f}")
#     return rewards

# def predict_ebm_reward(ebm_scorer, batch_features):
#     """
#     使用训练好的EBM模型，对一批样本计算奖励（每个样本独立输出概率）。
#     返回形状: [N] 的 1D Tensor，值域 [0,1]
#     """
#     ebm_model = ebm_scorer['model']
#     mean_feature = ebm_scorer['mean_feature']

#     if batch_features.device != mean_feature.device:
#         mean_feature = mean_feature.to(batch_features.device)

#     # 差分特征
#     diff_feature = batch_features - mean_feature  # [N, D]

#     # 转 numpy (EBM 通常是 sklearn 类型)
#     diff_feature_np = diff_feature.cpu().numpy()  # [N, D]

#     # 预测正类概率 (每个样本一个概率)
#     proba = ebm_model.predict_proba(diff_feature_np)  # [N, 2]
#     rewards = proba[:, 1]  # 取正类概率

#     # 转回 torch.Tensor
#     return torch.from_numpy(rewards).float()

def predict_ebm_reward(ebm_scorer, batch_features):
    """
    使用训练好的 EBM 模型预测每个样本的奖励分数。
    返回 reward ∈ [-1, 1]，每个样本一个。
    """
    import torch
    import numpy as np

    ebm_model = ebm_scorer['model']
    mean_feature = ebm_scorer['mean_feature']

    # 确保设备一致
    if batch_features.device != mean_feature.device:
        mean_feature = mean_feature.to(batch_features.device)

    # 1️⃣ 差分特征 Δx = x - μ（μ 为全局均值）
    diff_feature = batch_features - mean_feature

    # 2️⃣ 转为 numpy（保持批量维度）
    diff_feature_np = diff_feature.cpu().numpy()  # shape [N, D]

    # 3️⃣ 预测概率 P(x 胜 μ)
    proba = ebm_model.predict_proba(diff_feature_np)[:, 1]  # shape [N]

    # 4️⃣ 转换为 tensor 并归一化到 [-1, 1]
    reward = torch.from_numpy(proba).float()
    reward = 2 * (reward - 0.5)  # [0,1] → [-1,1]
    reward = torch.clamp(reward, -1.0, 1.0)

    # ✅ 返回每个样本的独立 reward 向量
    return reward.cuda() if batch_features.is_cuda else reward


# def predict_ebm_reward(ebm_scorer, batch_features):
#     """
#     使用训练好的EBM模型，通过与全局均值比较来预测奖励。

#     :param ebm_scorer: 包含 'model' 和 'mean_feature' 的字典。
#     :param batch_features: 单个批次的特征向量 (torch.Tensor)。
#     :return: 代理奖励分数 (float, 0到1之间)。
#     """
#     ebm_model = ebm_scorer['model']
#     mean_feature = ebm_scorer['mean_feature']

#     # 确保在同一设备上
#     if batch_features.device != mean_feature.device:
#         mean_feature = mean_feature.to(batch_features.device)

#     # 计算差分特征 Δx = x - μ
#     diff_feature = (batch_features - mean_feature)
#     # diff_feature = (diff_feature - diff_feature.mean(dim=0)) / (diff_feature.std(dim=0) + 1e-6)

#     # EBM需要numpy输入
#     diff_feature_np = diff_feature.cpu().numpy().reshape(1, -1)

#     # 预测 P(x 胜 μ)，即正类的概率
#     proba = ebm_model.predict_proba(diff_feature_np)
#     rewards = proba[:, 1]  # 取正类概率

#     return reward

