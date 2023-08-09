# AOCI: An Adaptive Object-conditional Causal Invariant Framework for Domain Generalization

PyTorch implementation of "An Adaptive Object-conditional Causal Invariant Framework for Domain Generalization". Our work is built based on DNA: Domain Generalization with Diversified Neural Averaging

## Usage
1. Dependencies
```sh
pip install -r requirements.txt
```

2. Download the datasets
```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```
3. Run training


For example, you can run the following instructions with different random dataset splits on PACS.(default params)
```
python train_all.py PACS0 --dataset PACS --deterministic  --data_dir ./dataset

```
The results are reported as a table. In the table, the row `SWAD` indicates out-of-domain accuracy of the ensemble model, and the row `SWAD(inD)` indicates the in-domain validation accuracy.

To reproduce the results of AOCI, we list the recommended hyperparameters searched by us in hparams_registry.py. You can also manually search hyperparameters by modifying them in CLI. 
## Requirements

Environment details used for our experiments.

```
Python: 3.7.9
PyTorch: 1.7.1
Torchvision: 0.8.2
CUDA: 11.2
CUDNN: 7605
NumPy: 1.19.4
PIL: 8.0.1
```






