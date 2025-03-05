# FSCfold: A Deep Learning Approach for Accurate Prediction of Pseudoknot-Containing RNA Secondary Structures Across Families

## Prepare for Experiments

Let us spend some time configuring the environment that FSCfold needs.

### Configure conda environment

We provide `environment.txt` files including all the environments that FSCfold depends on.

```bash
conda create -n FSCfold python=3.11 -y
conda activate FSCfold
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Data Download

All data used in the experiments have been shared to [Google Drive](https://drive.google.com/drive/folders/1JB059ESPbN2QISLZpjOJbRengrc1jNh6?usp=drive_link).

Place all test sets and txt files with predictions downloaded from Google Drive into the Dataset folder.


### Model Evaluating 
According to the paper, we have placed three `.pt` files in the Models folder. The `RNAStralign.pt` file was trained using the RNAStralign training set, and the results that can be validated include the RNAStralign test set from our paper, the ArchiveII dataset, and the RNAStralign_test_pseu test set containing pseudoknots.

The `TS0.pt` file was trained using the TS0 training set and tested on the TS0 test set.

Finally, to evaluate the cross-family prediction performance of FSCfold, we trained on all datasets except the bpnew dataset to obtain the model parameters for `bpnew.pt` and then tested using the bpnew dataset.
#### Testing of Model Parameters for RNAStralign.pt

```bash
python FSCfold_test.py --test_files ArchiveII # Testing the ArchiveII Dataset
python FSCfold_test.py --test_files RNAStralign # Testing the RNAStralign Dataset
python FSCfold_test.py --test_files RNAStralign_test_pseu # Testing the Pseudoknot  Dataset
```

#### Testing of Model Parameters for TS0.pt

```bash
python FSCfold_test.py --test_files TS0 # Testing the ts0 Dataset
```

#### Testing of Model Parameters for bpnew.pt

```bash
python FSCfold_test.py --test_files bpnew # Testing the bpnew Dataset
```

### Model Predicting
We welcome you to test our model's predictions. First, you can modify `MODEL_SAVED = Models/bpnew.pt` in the `FSCfold_predict.py` folder to replace it with the training model you would like to use. Additionally, place the one-dimensional RNA sequences you wish to predict into the `input.txt` file in the Dataset folder. The file currently contains three test single-stranded RNA sequences from our paper, and you can directly execute the command below to make predictions. Moreover, you can also store your sequences in this file, and predictions will still be generated. Our model supports parallel predictions for multiple RNA sequences.he prediction results are stored in `.ct` file format in the `Result/save_ct_file` folder. You can convert them to your desired format using `ct2bpseq.py` and `ct2fasta.py` located in the Result folder.
```bash
python FSCfold_predict.py. # Predicting RNA seq
```

### Model DIY Training
We also provide the model training method. You can create your own training data and generate your `pickle` or `cPickle` files to place them in the Dataset folder. Then, in the `Tool\utill.py` file, add your training set to the `--train_file` parameter within the `get_args` function. The model also incorporates a multi-dataset fusion parallel training method, allowing you to specify multiple parameters for different training sets separated by spaces for your DIY training. Below are the commands to train the model:
```bash
python FSCfold_train.py --train_files Your_dataset_A Your_dataset_B 
--train_files: optinal parameter, default is all the datasets mentioned in the paper.
```