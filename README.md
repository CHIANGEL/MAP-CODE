# MAP-CODE
Official Code for paper "MAP: A Model-agnostic Pretraining Framework for Click-through Rate Prediction"

**NOTE**: I have deleted some unrelated codes which are our preliminary exploratory experiments. If readers got any problems or come across any bugs, please kindly leave me a message.

## Requirement

```
pip install -r requirements.txt
```

## Data Preprocessing

We provide the data preprocessing scripts in ```data_preprocess``` folder. One can also download the preprocessed data from [[Link]]() and place it at the main folder.

## Quick Start

We provide demo scripts in ```run_script``` folder.

To train DCNv2 from scratch:
```
CUDA_VISIBLE_DEVICES=0 bash run_script/run_DCNv2_scratch.sh
```

To pretrain DCNv2 with MFP:
```
CUDA_VISIBLE_DEVICES=0 bash run_script/run_DCNv2_MFP.sh
```

To pretrain DCNv2 with RFD:
```
CUDA_VISIBLE_DEVICES=0 bash run_script/run_DCNv2_RFD.sh
```

To finetune DCNv2 after pretraining:
```
CUDA_VISIBLE_DEVICES=0 bash run_script/run_DCNv2_finetune.sh
```

## Citation

```
@inproceedings{lin2023map,
  title={MAP: A Model-agnostic Pretraining Framework for Click-through Rate Prediction},
  author={Lin, Jianghao and Qu, Yanru and Guo, Wei and Dai, Xinyi and Tang, Ruiming and Yu, Yong and Zhang, Weinan},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1384--1395},
  year={2023}
}
```