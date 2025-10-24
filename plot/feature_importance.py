import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# --- 中文字体设置（可选） ---
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False


def merge_features(data):
    """合并相关特征"""
    merge_map = {
        'temporal_inconsistency': [
            'mean_temporal_inconsistency',
            'std_temporal_inconsistency'
        ],
        'similarity': [
            'mean_similarity',
            'std_similarity'
        ],
        'entropy': [
            'mean_entropy',
            'std_entropy'
        ],
        'dist_to_labeled': [
            'std_dist_to_labeled',
            'mean_dist_to_labeled'
        ],
        'neighborhood_density': ['neighborhood_density_score'],
        'diversity': ['diversity_score'],
        'representativeness': ['representativeness_score']
    }

    merged_data = {}
    processed_keys = set()

    for new_name, original_names_list in merge_map.items():
        importance_sum = 0.0
        for old_name in original_names_list:
            if old_name in data:
                importance_sum += data[old_name]
                processed_keys.add(old_name)
        merged_data[new_name] = importance_sum

    for key, value in data.items():
        if key not in processed_keys:
            clean_key = key.replace('_score', '')
            merged_data[clean_key] = value

    return merged_data


# --- 原始数据 ---
ebm_feature_importance = {
    'neighborhood_density_score': 0.695308,
    'std_dist_to_labeled': 0.586904,
    'mean_similarity': 0.468236,
    'mean_entropy': 0.303403,
    'mean_temporal_inconsistency': 0.238909,
    'std_temporal_inconsistency': 0.192505,
    'std_similarity': 0.182248,
    'std_entropy': 0.179355,
    'mean_dist_to_labeled': 0.145726,
    'diversity_score': 0.138727,
    'representativeness_score': 0.121345
}

# --- 合并特征 ---
merged_data = merge_features(ebm_feature_importance)

# --- 数据准备 ---
labels = list(merged_data.keys())
values = np.array(list(merged_data.values()))

# --- 🔹 对数缩放（指数底数优化） ---
# 添加一个偏移避免 log(0)
values = np.log10(values * 50)  # *10是为了增强分辨率
print(values)
# --- 归一化 ---
#values = (values - values.min()) / (values.max() - values.min())

# --- 🔹 调整顺序：交换 temporal_inconsistency 与 representativeness ---
swap_a = labels.index("temporal_inconsistency")
swap_b = labels.index("representativeness")
labels[swap_a], labels[swap_b] = labels[swap_b], labels[swap_a]
values[swap_a], values[swap_b] = values[swap_b], values[swap_a]

# --- 闭合数据 ---
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
values = np.concatenate((values, [values[0]]))
angles += angles[:1]

# --- 绘图 ---
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#f9fafb")

# 填充与线条
ax.fill(angles, values, color='skyblue', alpha=0.4)
ax.plot(angles, values, color='darkblue', linewidth=2)
ax.scatter(angles, values, color='navy', s=40, zorder=3)

# --- 标签 ---
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12, color='black')
ax.tick_params(axis='x', pad=10)

# --- 隐藏坐标数字 ---
ax.set_yticklabels([])
ax.set_yticks([])

# --- 视觉优化 ---
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.spines['polar'].set_color('#999')
ax.grid(color='#ccc', linestyle='--', linewidth=0.8)
plt.title("Feature Importance Radar Chart", fontsize=18, pad=30, color='black')

plt.tight_layout()
plt.show()
