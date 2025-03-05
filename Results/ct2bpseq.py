def ct_to_bpseq(ct_file_path, bpseq_file_path):
    with open(ct_file_path, 'r') as ct_file:
        lines = ct_file.readlines()

    with open(bpseq_file_path, 'w') as bpseq_file:
        for line in lines:
            if line.startswith('>') or line.strip() == "":
                continue  # 跳过注释行和空行
            parts = line.split()
            if len(parts) < 6:
                continue  # 如果行不符合.ct文件格式，则跳过
            index = parts[0]
            base = parts[1]
            pair = parts[4]  # 获取配对信息
            if pair == '0':
                pair = '0'  # 如果没有配对，则配对位置为0
            bpseq_file.write(f"{index} {base} {pair}\n")

ct_file_path = '_RNAStrAlign_5S_rRNA_database_Bacteria_B06883.ct.ct'
bpseq_file_path = '_RNAStrAlign_5S_rRNA_database_Bacteria_B06883.ct.bpseq'

ct_to_bpseq(ct_file_path, bpseq_file_path)
