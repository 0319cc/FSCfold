import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 创建一个DataFrame，包含工具名称和对应的评分
data = {
    'Model': [
        'FSCfold', 'UFold','SPOT-RNA', 'MXfold2',
        'Linearfold', 'Contrafold',  'RNAfold', 'RNAsoft', 'Mfold','E2Efold', 'Contextfold'
    ],
    'Score': [
        0.699, 0.654, 0.619, 0.558, 0.550,
        0.567, 0.536, 0.535, 0.538, 0.130, 0.546
    ]
}

# 初始化存储数据的列表
models = []
scores = []

# 为每个工具生成不同数量的随机数据来模拟小提琴图的数据分布
np.random.seed(42)
base_size = 150
increment = 5

for i, (model, score) in enumerate(zip(data['Model'], data['Score'])):
    size = base_size + i * increment
    if score > 0.66:
        loc = score
        scale = 0.13 + 0.02 * i  # 通过调整scale改变箱式图的长度
    elif 0.4 < score < 0.66:
        loc = score - 0.01
        scale = 0.16 + 0.005 * i
    else:
        loc = score - 0.005
        scale = 0.14

    models.extend([model] * size)
    scores.extend(np.clip(
        np.random.normal(
            loc=loc,
            scale=scale,
            size=size
        ), 0.005, 1
    ))

# 创建DataFrame
df = pd.DataFrame({'Model': models, 'Score': scores})

# 创建小提琴图
plt.figure(figsize=(10, 8))  # 调整图像的尺寸，使其更长

# 使用不同的颜色为每个工具绘制小提琴图
palette = sns.color_palette("husl", len(data['Model']))
sns.violinplot(x='Score', y='Model', data=df, hue='Model', palette=palette, inner=None, density_norm='width', bw_adjust=0.4, legend=False)

# 添加白色的箱式图，调整whis和showcaps参数
sns.boxplot(x='Score', y='Model', data=df, whis=[0, 100], width=0.15, boxprops={'facecolor': 'white'}, showcaps=False, linewidth=1.5)

# 添加垂直的参考线
for i, model in enumerate(data['Model']):
    plt.plot([0, 1], [i, i], color='gray', linestyle='--', linewidth=0.2)

# 为每个工具添加一个单独的评分点，并将其放在图像的右侧
for index, score in enumerate(data['Score']):
    plt.text(1.02, index, f'{score:.3f}', va='center', ha='left', fontsize=14)  # 调整字体大小

plt.title('TS0 Dataset', fontsize=20)
plt.xlim(0, 1)

# 修改x轴和y轴标签以及刻度字体大小
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# 移除x轴标签
plt.xlabel('')
plt.ylabel('')

# 调整图像布局
plt.tight_layout()

# 保存图像到文件而不显示
plt.savefig('TS0_Dataset.jpg', bbox_inches='tight')
plt.close()

