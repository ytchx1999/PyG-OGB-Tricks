# Residual Network and Embedding Usage: New Tricks of Node Classification with Graph Convolutional Networks

[![](https://img.shields.io/badge/CSDN-@智慧的旋风-red.svg?style=plastic)](https://blog.csdn.net/weixin_41650348/)  [![img](https://img.shields.io/badge/contributions-welcome-success.svg?style=plastic&logo=Github)](https://github.com/ytchx1999/PyG-OGB-Tricks) ![GitHub](https://img.shields.io/github/license/ytchx1999/PyG-OGB-Tricks?logo=Github&style=plastic) ![GitHub last commit](https://img.shields.io/github/last-commit/ytchx1999/PyG-OGB-Tricks?logo=github&style=plastic) ![GitHub Repo stars](https://img.shields.io/github/stars/ytchx1999/PyG-OGB-Tricks?style=social) ![GitHub forks](https://img.shields.io/github/forks/ytchx1999/PyG-OGB-Tricks?style=social)  

**[Huixuan Chi](https://scholar.google.com.hk/citations?hl=zh-CN&user=mSLoo54AAAAJ), Yuying Wang, Qinfen Hao, Hong Xia**

Bags of Tricks in OGB (node classification) with GCNs.

In this work, we propose two novel tricks of GCNs for node classification tasks:  **GCN\_res Framework** and **Embedding Usage**, which can improve various GCNs significantly. Experiments on [Open Graph Benchmark](https://ogb.stanford.edu/)  (OGB) show that, by combining these techniques, the test accuracy of various GCNs increases by **1.21**%~**2.84**%. 

Our paper is available at https://arxiv.org/abs/2105.08330.

My [blog](https://blog.csdn.net/weixin_41650348/article/details/113916600?spm=1001.2014.3001.5501) records the detailed ranking process on OGB.

## *Overview*

#### `ogbn-arxiv`

+ **Code:** [https://github.com/ytchx1999/PyG-ogbn-arxiv](https://github.com/ytchx1999/PyG-ogbn-arxiv)
+ [**Leaderboard**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv)

#### `ogbn-mag`

+ **Code:** [https://github.com/ytchx1999/PyG-ogbn-mag](https://github.com/ytchx1999/PyG-ogbn-mag)
+ [**Leaderboard**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag)

#### `ogbn-products`

+ **Code:** [https://github.com/ytchx1999/PyG-ogbn-products](https://github.com/ytchx1999/PyG-ogbn-products)
+ [**Leaderboard**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-products)

#### `ogbn-proteins`

+ **PyG-Code:** [https://github.com/ytchx1999/PyG-ogbn-proteins](https://github.com/ytchx1999/PyG-ogbn-proteins)
+ **DGL-Code:** [https://github.com/ytchx1999/PyG-OGB-Tricks/tree/main/DGL-ogbn-proteins](https://github.com/ytchx1999/PyG-OGB-Tricks/tree/main/DGL-ogbn-proteins)
+ [**Leaderboard**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-proteins)

## *Methods*

### GCN_res Framework

<img src="./image/GCN_res_fig.jpeg" style="zoom:20%;" />

Overview of GCN_res Framework with a 4-layer toy example. The GCNs-Block consists of four parts: GCNsConv layer, Norm layer, activation function, and Dropout unit. Data stream of residual connections is indicated by arrows.

In this paper, we propose GCN res Framework by two main strategies in the forward propagation: (i) adaptive residual connections and initial residual connections; and (ii) softmax layer-aggregation.

### Embedding Usage

<img src="./image/Embedding.jpeg" alt="Embedding Usage" style="zoom:24%;" />

Embedding Usage for GCNs. We merge input featrues with embedding to generate new features for GCNs.

In this work, we take an initial step towards answering the questions above by proposing Embedding Usage to enhance node features.

## *Results on OGB Datasets*

###  Requirements

```python
pytorch >= 1.6.0
torch-geometric >= 1.6.0
dgl >= 0.5.0
ogb >= 1.1.1
```

### `ogbn-arxiv`

| **Model**                      | **Test(%)** | **Valid(%)** |
| :--------------- | :---------: | :----------: |
| MLP                            |    55.50 ± 0.23    |    57.65 ± 0.12    |
| GCN (3)                        |    71.74 ± 0.29    |    73.00 ± 0.17    |
| GCN + FLAG (3)                 |    72.04 ± 0.20    |    73.30 ± 0.10    |
| SIGN                           |    71.95 ± 0.11    |    73.23 ± 0.06    |
| DeeperGCN                      |    71.92 ± 0.16    |    72.62 ± 0.14    |
| DAGNN                          |    72.09 ± 0.25    |    72.90 ± 0.11    |
| JKNet                          |    72.19 ± 0.21    |    73.35 ± 0.07    |
| GCNII                          |    72.74 ± 0.16    |      —       |
| UniMP                          |    73.11 ± 0.20    |  **74.50 ± 0.05**  |
| **GCN\_res** (8)               |    72.62 ± 0.37    |    73.69 ± 0.21    |
| **GCN\_res** + **FLAG** (8)    |    72.76 ± 0.24    |    73.89 ± 0.12    |
| **GCN\_res** + **C&S\_v2** (8) |    73.13 ± 0.17    |    74.45 ± 0.11    |
| **GCN\_res** + **C&S\_v3** (8) |  **73.91 ± 0.14**  |    73.61 ± 0.21    |
| GAT + BoT                         |    73.91 ± 0.12   |    **75.16 ± 0.08**    |
| **GAT-node2vec** + **BoT**  |  74.05 ± 0.04  |    74.82 ± 0.15    |
| **GAT-node2vec** + **BoT** + **self-KD**  |  **74.20 ± 0.04**  |    74.82 ± 0.15    |

### `ogbn-mag`

| **Model**                                    |   **Test(%)**    |   **Valid(%)**   |
| :------------------------------------------- | :--------------: | :--------------: |
| GraphSAINT (R-GCN aggr)                      |   47.51 ± 0.22   |   48.37 ± 0.26   |
| **GraphSAINT** + **metapath2vec**            |   49.66 ± 0.22   |   50.66 ± 0.17   |
| **GraphSAINT** + **metapath2vec** + **C&S**  |   48.43 ± 0.24   |   49.36 ± 0.24   |
| **GraphSAINT** + **metapath2vec** + **FLAG** | **49.69 ± 0.22** | **50.88 ± 0.18** |
| R-GSN                                        |   50.32 ± 0.37   |   51.82 ± 0.41   |
| **R-GSN** + **metapath2vec**                 | **51.09 ± 0.38** | **52.95 ± 0.42** |

### `ogbn-products`

| **Model**                                            |   **Test(%)**    |   **Valid(%)**   |
| :--------------------------------------------------- | :--------------: | :--------------: |
| Full-batch GraphSAGE                                 |   78.50 ± 0.14   |   92.24 ± 0.07   |
| GraphSAGE w/NS                                       |   78.70 ± 0.36   |   91.70 ± 0.09   |
| GraphSAGE w/NS + FLAG                                |   79.36 ± 0.57   |   92.05 ± 0.07   |
| **GraphSAGE w/NS** + **BN** + **C&S**                |   80.41 ± 0.22   |   92.38 ± 0.07   |
| **GraphSAGE w/NS** + **BN** + **C&S** + **node2vec** | **81.54 ± 0.50** | **92.38 ± 0.06** |

### `ogbn-proteins`

| **Model**                         |   **Test(%)**    |   **Valid(%)**   |
| :-------------------------------- | :--------------: | :--------------: |
| **GEN**                           |   81.30 ± 0.65   |   85.74 ± 0.53   |
| **GEN** + **FLAG**                |   81.29 ± 0.67   |   85.87 ± 0.54   |
| **GEN** + **FLAG** + **node2vec** | **82.51 ± 0.43** | **86.56 ± 0.37** |
| GAT             |   86.82 ± 0.21   |   91.94 ± 0.03   |
| **GAT** + **labels** + **node2vec**                |   **87.11 ± 0.07**   |   **92.17 ± 0.11**   |

### t-SNE visualization on `ogbn-arxiv`

<img src="./image/t-SNE.png" alt="t-" style="zoom:60%;" />

## *Cite*

Please cite our paper if you find anything helpful,

```bibtex
@article{chi2021residual,
  title={Residual Network and Embedding Usage: New Tricks of Node Classification with Graph Convolutional Networks},
  author={Chi, Huixuan and Wang, Yuying and Hao, Qinfen and Xia, Hong},
  journal={arXiv preprint arXiv:2105.08330},
  year={2021}
}
```

