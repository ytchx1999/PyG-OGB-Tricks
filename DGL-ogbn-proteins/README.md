# GAT + labels + node2vec
This is an improvement of the [GAT](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-proteins) model by Wang (DGL), using node2vec embedding. 

Our paper is available at [https://arxiv.org/pdf/2105.08330.pdf](https://arxiv.org/pdf/2105.08330.pdf).

### ogbn-proteins

#### Improvement Strategy：

+ adjust hidden and embedding dim.
+ add node2vec embedding ---- the usage of node2vec greatly accelerates the convergence of GAT.

#### Environmental Requirements

+ dgl >= 0.5.0
+ torch >= 1.6.0
+ torch_geometric >= 1.6.0
+ ogb == 1.3.0

#### Experiment Setup：

1. Generate node2vec embeddings, which save in `proteins_embedding.pt`

   ```bash
   python node2vec_proteins.py
   ```

2. Run the real model

   + **Let the program run in the foreground.**

   ```bash
   python gat.py --use-labels
   ```

   + **Or let the program run in the background** and save the results to a log file.

   ```bash
   nohup python gat.py --use-labels > ./gat.log 2>&1 &
   ```

#### Detailed Hyperparameter:

GAT:

```bash
Namespace(attn_drop=0.0, cpu=False, dropout=0.25, edge_drop=0.1, eval_every=5, gpu=0, input_drop=0.1, log_every=5, lr=0.01, n_epochs=1200, n_heads=6, n_hidden=128, n_layers=6, n_runs=10, no_attn_dst=False, plot_curves=False, save_pred=False, seed=0, use_embed=True, use_labels=True, wd=0)

--n-runs N_RUNS         running times (default: 10)
--n-epochs N_EPOCHS     number of epochs (default: 1200)
--use-labels            Use labels in the training set as input features. (default: False)
--lr LR                 learning rate (default: 0.01)
--n-layers N_LAYERS     number of layers (default: 6)
--n-heads N_HEADS       number of heads (default: 6)
--n-hidden N_HIDDEN     number of hidden units (default: 128)
--dropout DROPOUT       dropout rate (default: 0.25)
--input-drop INPUT_DROP input drop rate (default: 0.1)
```

node2vec:

```bash
embedding_dim = 16
lr = 0.01
batch_size = 256
walk_length = 80
epochs = 5
```

#### Result:

```python
Val scores: [0.9229285934246892, 0.9211608885028892, 0.9213509308888836, 0.9219311666881109, 0.922188157691978, 0.9233155178378067, 0.9226761093114175, 0.9207967425451954, 0.9192225312946334, 0.9216411187053957]
Test scores: [0.8705177963169082, 0.8718678325708628, 0.871026339976343, 0.8713582109483052, 0.8706036035560922, 0.8709027982169764, 0.8704158483168263, 0.8704708862546975, 0.8713362807645616, 0.8726814140948117]

Average val score: 0.9217211756890998 ± 0.0011282315196969204
Average test score: 0.8711181011016385 ± 0.0006857984340481437
```

| Model                   | Test Accuracy   | Valid Accuracy  | Parameters | Hardware          |
| ----------------------- | --------------- | --------------- | ---------- | ----------------- |
| GAT + labels + node2vec | 0.8711 ± 0.0007 | 0.9217 ± 0.0011 | 6360470    | Tesla V100 (32GB) |

