# Unsupervised Visible-Infrared Cross-Modal Person Re-Identification via Dual-modality Data Augmentation and Adaptive Counterfactual Reasoning Learning


## Dataset Preprocessing
Convert the dataset format (like RegDB).
```shell
python prepare_sysu.py   # for SYSU-MM01
python prepare_regdb.py  # for RegDB
```
You need to change the file path in the `prepare_sysu(regdb).py`.


## Training
```shell
bash train_sysu.sh   # for SYSU-MM01
bash train_regdb.sh  # for RegDB
```

## Test
```shell
bash test_sysu.sh    # for SYSU-MM01
bash test_regdb.sh   # for RegDB
```


