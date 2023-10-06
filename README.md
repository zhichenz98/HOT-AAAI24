# Hierarchical Multi-Marginal Optimal Transport for Network Alignment

## Overview

**prerequisites**
- pot>=0.9.0
- numpy>=1.22.4
- scikit-learn>=1.1.2

**files**
- ```./src/```: source code to reproduce the experiments
    - ```hot_utils```: utilities for hot, including calculating rwr, cross-graph cost, intra-graph cost, etc.
    -  ```hot```: the main HOT algorithm
    -  ```log_mot```: multi-marginal optimal transport in the log domain
    - ```run_ACM```: Experiments on ACM and ACM(A) datasets
    - ```run_DBLP```: Experiments on DBLP and DBLP(A) datasets
    - ```run_douban```: experiments on Douban-230 dataset
- ```./dataset/```: dataset used for experiments

## How to use
Simply run the ```run_ACM.py```, ```run_DBLP.py```, and ```run_douban.py``` to reproduce the experiment results.
