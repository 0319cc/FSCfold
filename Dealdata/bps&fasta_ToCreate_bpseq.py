import os

# 获取当前脚本所在目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# TS0 文件夹路径
ts0_folder_path = os.path.join(current_dir, "TS1")

# 创建 TS0_bpseq 文件夹，如果它不存在
ts0_bpseq_folder_path = os.path.join(current_dir, "TS1_bpseq")
os.makedirs(ts0_bpseq_folder_path, exist_ok=True)

# 遍历 TS0 文件夹中的文件
for filename in os.listdir(ts0_folder_path):
    # 只处理 .bps 文件
    if filename.endswith(".bps"):
        bps_filename = os.path.join(ts0_folder_path, filename)
        fasta_filename = os.path.join(ts0_folder_path, filename.replace(".bps", ".fasta"))
        bpseq_filename = os.path.join(ts0_bpseq_folder_path, filename.replace(".bps", ".bpseq"))

        # 确保对应的 FASTA 文件存在
        if os.path.exists(fasta_filename):
            # 读取 BPS 文件
            bps_data = []
            with open(bps_filename, "r") as bps_file:
                # 跳过标题行
                next(bps_file)
                for line in bps_file:
                    try:
                        i, j = line.strip().split()
                        bps_data.append((int(i), int(j)))
                    except ValueError:
                        print(f"跳过无效行: {line.strip()}")
                        continue

            # 读取 FASTA 文件
            fasta_data = ""
            with open(fasta_filename, "r") as fasta_file:
                for line in fasta_file:
                    if not line.startswith(">"):
                        fasta_data += line.strip()

            # 生成 BPSEQ 文件
            with open(bpseq_filename, "w") as bpseq_file:
                for index, (i, j) in enumerate(bps_data):
                    base = fasta_data[index]
                    bpseq_file.write(f"{i} {base} {j}\n")

            print(f"BPSEQ 文件生成完毕：{bpseq_filename}")
        else:
            print(f"对应的 FASTA 文件未找到：{fasta_filename}")

print("所有 BPSEQ 文件生成完毕。")
