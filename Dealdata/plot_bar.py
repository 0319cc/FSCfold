import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 原始数据
length_intervals_original = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550,
                             600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150,
                             1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750]
sequences_original = [7, 6491, 9435, 18, 247, 638, 656, 913, 634, 416, 342, 311,
                      171, 164, 175, 352, 578, 214, 131, 109, 52, 52, 37, 74,
                      128, 104, 388, 1004, 2272, 2989, 1195, 142, 2, 5, 4, 1]

# 汇总数据
length_intervals = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
sequences = [
    sum(sequences_original[0:4]),  # 0-200
    sum(sequences_original[4:8]),  # 200-400
    sum(sequences_original[8:12]), # 400-600
    sum(sequences_original[12:16]), # 600-800
    sum(sequences_original[16:20]), # 800-1000
    sum(sequences_original[20:24]), # 1000-1200
    sum(sequences_original[24:28]), # 1200-1400
    sum(sequences_original[28:32]), # 1400-1600
    sum(sequences_original[32:36])  # 1600-1800
]

# 使用一个色调相近的色谱
colors = cm.viridis([i/len(length_intervals) for i in range(len(length_intervals))])

# 绘制条形图
plt.figure(figsize=(14, 7))
bars = plt.bar(length_intervals, sequences, color=colors, width=200, align='edge')
plt.xlabel('RNA sequence length', fontsize=16)  # 设置x轴标签字体大小
plt.ylabel('Number of Sequences', fontsize=16)  # 设置y轴标签字体大小
# plt.title('Number of Sequences in Different Length Intervals')
plt.xticks(range(0, 1801, 200))  # 设置x轴刻度间隔为200，从0到1800
plt.yticks(range(0, 17501, 2500))  # 设置y轴刻度间隔为2500，从0到17500

# 去掉x轴上的小刻度线
plt.gca().tick_params(axis='x', which='minor', bottom=False)

# 设置轴范围
plt.xlim(0, 1800)
plt.ylim(0, 17500)

# 在每个柱形上面标注数值
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', ha='center', va='bottom')

# 保存图表为JPG文件
plt.savefig('sequences_bar_chart.jpg', format='jpg', dpi=300)

# 显示图表
plt.show()
