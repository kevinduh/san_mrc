# Stochastic Answer Networks for Machine Reading Comprehension

This PyTorch package implements the Stochastic Answer Network (SAN) for Machine Reading Comprehension, as described in:

Xiaodong Liu, Yelong Shen, Kevin Duh, Jianfeng Gao<br/>
Stochastic Answer Networks for Machine Reading Comprehension<br/>
Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)<br/>
[arXiv version](https://arxiv.org/abs/1712.03556)


Xiaodong Liu, Wei Li, Yuwei Fang, Aerin Kim, Kevin Duh and Jianfeng Gao<br/>
Stochastic Answer Networks for SQuAD 2.0 <br/>
Technical Report
[arXiv version](https://arxiv.org/abs/1809.09194)


Please cite the above papers if you use this code. 

## Quickstart 

### Setup Environment
1. python3.6
2. install requirements:
   > pip install -r requirements.txt
3. download data/word2vec 
   > sh download.sh
4. You might need to download the en module for spacy
   > python -m spacy download en              # default English model (~50MB) <br/>
   > python -m spacy download en_core_web_md  # larger English model (~1GB)

### Train a SAN Model on SQuAD v1.1
1. preprocess data
   > python prepro.py
2. train a model
   > python train.py

### Train a SAN Model on SQuAD v2.0
1. preprocess data
   > python prepro.py --v2_on --train_data squad_train_v2.json --dev_data squad_dev_v2.json --meta squad_meta_v2.pick
2. train a Model
   > python train.py --data_dir data --train_data squad_train_v2.json --dev_data squad_dev_v2.json --dev_gold data\dev-v2.0.json --meta squad_meta_v2.pick --v2_on

## TODO
1. Add ELMo.
2. Add Multi-Task Training

## Notes and Acknowledgments
Some of code are adapted from: https://github.com/hitvoice/DrQA

## Results
We report results produced by this package as follows.

| Dataset | EM/F1 on Dev |
| ------- | ------- |
| `SQuAD v1.1` (Rajpurkar et al., 2016) | **76.8**/**84.6** (vs 76.2/84.1 SAN paper) |
| `SQuAD v2.0`  (Rajpurkar et al., 2018)| **69.5**/**72.7** |
| `NewsQA` (Trischler et al., 2016)| **55.8**/**67.9**|

Related:
1. <a href="https://arxiv.org/abs/1809.06963">Multi-Task Learning for MRC</a>
2. <a href="https://arxiv.org/abs/1804.07888">NLI</a>


