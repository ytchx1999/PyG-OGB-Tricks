# Residual Network and Embedding Usage: New Tricks of Node Classification with Graph Convolutional Networks

Bags of Tricks in OGB (node classification) with GCNs.

In this work, we propose two novel tricks of GCNs for node classification tasks:  **GCN\_res Framework** and **Embedding Usage**, which can improve various GCNs significantly. Experiments on [Open Graph Benchmark](https://ogb.stanford.edu/)  (OGB) show that, by combining these techniques, the test accuracy of various GCNs increases by **1.21**%$\sim$**2.84**%. 

## *Overview*

Paper[].

#### `ogbn-arxiv`

+ **Code:** [**https://github.com/ytchx1999/PyG-ogbn-arxiv**](https://github.com/ytchx1999/PyG-ogbn-arxiv)
+ **Leaderboard**: [**https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-arxiv)

#### `ogbn-mag`

+ **Code:** [**https://github.com/ytchx1999/PyG-ogbn-mag**](https://github.com/ytchx1999/PyG-ogbn-mag)
+ **Leaderboard**: [**https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-mag)

#### `ogbn-products`

+ **Code:** [**https://github.com/ytchx1999/PyG-ogbn-products**](https://github.com/ytchx1999/PyG-ogbn-products)
+ **Leaderboard**: [**https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-products**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-products)

#### `ogbn-proteins`

+ **Code:** [**https://github.com/ytchx1999/PyG-ogbn-proteins**](https://github.com/ytchx1999/PyG-ogbn-proteins)
+ **Leaderboard**: [**https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-proteins**](https://ogb.stanford.edu/docs/leader_nodeprop/#ogbn-proteins)

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

- PyTorch >= 1.6.0
- torch-geometric >= 1.6.0
- ogb >= 1.1.1

### `ogbn-arxiv`

| **Model**                      | **Test(%)** | **Valid(%)** |
| :--------------- | :---------: | :----------: |
| MLP                            |    55.50    |    57.65     |
| GCN (3)                        |    71.74    |    73.00     |
| GCN + FLAG (3)                 |    72.04    |    73.30     |
| SIGN                           |    71.95    |    73.23     |
| DeeperGCN                      |    71.92    |    72.62     |
| DAGNN                          |    72.09    |    72.90     |
| JKNet                          |    72.19    |    73.35     |
| GCNII                          |    72.74    |      —       |
| UniMP                          |    73.11    |  **74.50**   |
| **GCN\_res** (8)               |    72.62    |    73.69     |
| **GCN\_res** + **FLAG** (8)    |    72.76    |    73.89     |
| **GCN\_res** + **C&S\_v2** (8) |    73.13    |    74.45     |
| **GCN\_res** + **C&S\_v3** (8) |  **73.91**  |    73.61     |

### `ogbn-mag`

| **Model**                                    | **Test(%)** | **Valid(%)** |
| :------------------------------------------- | :---------: | :----------: |
| GraphSAINT (R-GCN aggr)                      |    47.51    |    48.37     |
| **GraphSAINT** + **metapath2vec**            |    49.66    |    50.66     |
| **GraphSAINT** + **metapath2vec** + **C&S**  |    48.43    |    49.36     |
| **GraphSAINT** + **metapath2vec** + **FLAG** |  **49.69**  |  **50.88**   |

### `ogbn-products`

| **Model**                                            | **Test(%)** | **Valid(%)** |
| :--------------------------------------------------- | :---------: | :----------: |
| Full-batch GraphSAGE                                 |    78.50    |    92.24     |
| GraphSAGE w/NS                                       |    78.70    |    91.70     |
| GraphSAGE w/NS + FLAG                                |    79.36    |    92.05     |
| **GraphSAGE w/NS** + **BN** + **C&S**                |    80.41    |    92.38     |
| **GraphSAGE w/NS** + **BN** + **C&S** + **node2vec** |  **81.54**  |  **92.38**   |

### `ogbn-proteins`

| **Model**                         | **Test(%)** | **Valid(%)** |
| :-------------------------------- | :---------: | :----------: |
| **GEN**                           |    81.30    |    85.74     |
| **GEN** + **FLAG**                |    81.29    |    85.87     |
| **GEN** + **FLAG** + **node2vec** |  **82.51**  |  **86.56**   |

### t-SNE visualization on `ogbn-arxiv`

<img src="./image/t-SNE.png" alt="t-" style="zoom:60%;" />

## *Cite*

Please cite our paper if you find anything helpful,

```

```

