import pandas as pd

# 示例数据
data = {
    'Model': ['FSCfold','GCNfold', 'UFold', 'MXfold2', 'E2Efold', 'SPOT-RNA', 'CONTRAfold', 'CDPfold', 'Linearfold', 'RNAStructure', 'RNAfold', 'Mfold'],
    'F1_Score': [0.982, 0.959, 0.955, 0.877, 0.821, 0.732, 0.633, 0.614, 0.609, 0.550, 0.540, 0.420]
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 保存为CSV文件
csv_file_path = 'RNAStralign_test.csv'
df.to_csv(csv_file_path, index=False)

print(f"数据已保存到 {csv_file_path}")
