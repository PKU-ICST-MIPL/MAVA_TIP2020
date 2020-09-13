# Introduction

This is the source code of our IEEE Transactions on Image Processing (TIP) 2020 paper "MAVA: Multi-level Adaptive Visual-textual Alignment by Cross-media Bi-attention Mechanism", Please cite the following paper if you find our code useful.

Yuxin Peng, Jinwei Qi and Yunkan Zhuo, "MAVA: Multi-level Adaptive Visual-textual Alignment by Cross-media Bi-attention Mechanism", IEEE Transactions on Image Processing (TIP), Vol. 29, No. 1, pp. 2728-2741, Dec. 2020. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201924)

# Preparation
Our code is based on [pytorch 1.0](https://pytorch.org/get-started), and tested on Ubuntu 14.04.5 LTS, Python 2.7.

# Usage

Data Preparation: we use flickr-30K as example, the data should be put in `./data/flickr`.

Run `sh ./scripts/run_global.sh` to train and test the global-level model.
Run `sh ./scripts/run_local.sh` to train and test the local-level model.
Run `sh ./scripts/run_relation.sh` to train and test the relation-level model.

Then run eval.m to evaluate the performance of multi-level model. 

# Our Related Work
If you are interested in cross-media retrieval, you can check our recently published paper:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol.28, No.9, pp.2372-2385, 2018. [[PDF]](http://59.108.48.34/tiki/download_paper.php?fileId=201823)

Welcome to our [Benchmark Website](http://59.108.48.34/tiki/XMediaNet/) and [Laboratory Homepage](http://mipl.icst.pku.edu.cn) for more information about our papers, source codes, and datasets.