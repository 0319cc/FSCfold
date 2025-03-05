def ct_to_fasta(ct_file, fasta_file):
    with open(ct_file, 'r') as ct:
        lines = ct.readlines()

    header = lines[0].strip()  # Get the header line
    sequence = ''.join([line.split()[1] for line in lines[1:]]).upper()  # Extract and concatenate the sequence

    with open(fasta_file, 'w') as fasta:
        fasta.write(f">{header}\n")
        fasta.write(f"{sequence}\n")

# 使用示例
ct_file_path = 'grp1_a.I1.e.L.dispersa.UNK.SSU.1046.ct'
fasta_file_path = 'grp1_a.I1.e.L.dispersa.UNK.SSU.1046.fasta'

ct_to_fasta(ct_file_path, fasta_file_path)
