import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 创建一个DataFrame，包含工具名称和对应的评分
data = {
    'Model': [
        'FSCfold','UFold','Contrafold','SPOT-RNA','MXfold2',
        'E2Efold','RNAsoft','Mfold','Linearfold',
        'Contextfold','RNAfold'

    ],
    'Score': [
        0.689, 0.636, 0.661, 0.620, 0.641,
        0.036,0.620, 0.623, 0.658, 0.604,0.640
    ]
}

# 为每个工具生成一些随机数据来模拟小提琴图的数据分布
np.random.seed(42)
df = pd.DataFrame({
    'Model': np.repeat(data['Model'], 100),
    'Score': np.concatenate([
        np.clip(np.random.normal(loc=score, scale=0.05, size=100), 0, 1)
        for score in data['Score']
    ])
})

# 创建小提琴图
plt.figure(figsize=(12, 8))  # 调整图像的尺寸，使其更长

# 使用不同的颜色为每个工具绘制小提琴图
palette = sns.color_palette("husl", len(data['Model']))
sns.violinplot(x='Score', y='Model', data=df, hue='Model', palette=palette, inner=None, density_norm='width', legend=False, bw_adjust=0.5)

# 添加白色的箱式图，调整whis和showcaps参数
sns.boxplot(x='Score', y='Model', data=df, whis=[0, 100], width=0.2, boxprops={'facecolor':'white'}, showcaps=False, linewidth=1.5)

# 添加垂直的参考线
for i, model in enumerate(data['Model']):
    plt.plot([0, 1], [i, i], color='gray', linestyle='--', linewidth=0.5)

# 为每个工具添加一个单独的评分点，并将其放在图像的右侧
for index, score in enumerate(data['Score']):
    plt.text(1.02, index, f'{score:.3f}', va='center', ha='left', fontsize=14)  # 调整字体大小

plt.title('bpRNA-new', fontsize=20)
plt.xlim(0, 1)

# 修改x轴和y轴标签以及刻度字体大小
plt.xlabel('F1 Score', fontsize=18)
plt.yticks(fontsize=13)

# 移除y轴标签
plt.ylabel('')

# 添加左边和下面的边框
# sns.despine(left=False, bottom=False)

# 调整图像布局
plt.tight_layout()

# 保存图像到文件而不显示
plt.savefig('bpRNA-new.jpg', bbox_inches='tight')
plt.close()
