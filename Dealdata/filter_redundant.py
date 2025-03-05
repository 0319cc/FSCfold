import warnings
import _pickle as cPickle
import numpy as np
import os
from os import walk
import pandas as pd
import collections
from collections import defaultdict

dataset = 'rnastralign'
rna_types = ['tmRNA', 'tRNA', 'telomerase', 'RNaseP',
    'SRP', '16S_rRNA', '5S_rRNA', 'group_I_intron']

datapath = './RNAStrAlign'
seed = 0

# Select all files within the preferred rna_type
file_list = list()

for rna_type in rna_types:
    type_dir = os.path.join(datapath, rna_type+'_database')
    for r, d, f in walk(type_dir):
        for file in f:
            if file.endswith(".ct"):
                file_list.append(os.path.join(r, file))

# Load data
data_list = list(map(lambda x: pd.read_csv(x, sep='\s+', skiprows=1, header=None), file_list))

seq_list = list(map(lambda x: ''.join(list(x.loc[:, 1])), data_list))

seq_file_pair_list = list(zip(seq_list, file_list))
d = defaultdict(list)
for k, v in seq_file_pair_list:
    d[k].append(v)
unique_seqs = list()
seq_files = list()
for k, v in d.items():
    unique_seqs.append(k)
    seq_files.append(v)

unique_seq_len = list(map(len, unique_seqs))

# Filter sequences with length <= 600
filtered_seq_len = list(filter(lambda x: x <= 2000, unique_seq_len))

# Suppress deprecation warnings for distplot
warnings.filterwarnings("ignore", category=FutureWarning)

# Count sequences with length <= 600
seq_count_600 = len(filtered_seq_len)
print(f'Total sequences with length <= 600: {seq_count_600}')

# Calculate histogram data for lengths
bins = range(0, 2001, 200)
hist, bin_edges = np.histogram(filtered_seq_len, bins=bins)

# Print out the counts for each interval
for i in range(len(hist)):
    print(f'Length interval {bin_edges[i]}-{bin_edges[i+1]}: {hist[i]} sequences')
