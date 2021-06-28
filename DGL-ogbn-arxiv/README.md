# GAT-node2vec + BoT + self-KD
This is another attempt at Tricks of GNNs. Our paper is available at [https://arxiv.org/pdf/2105.08330.pdf](https://arxiv.org/pdf/2105.08330.pdf).

We refer to the codes of [GAT+norm. adj.+label reuse](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv) by Wang (DGL) and [GAT+label reuse+self KD](https://github.com/ShunliRen/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv) by Ren. 

### ogbn-arxiv

#### Improvement Strategy：

+ Embedding Usage
  + Add node2vec embedding ---- the usage of node2vec greatly accelerates the convergence of GAT.
  + More details can be found in our paper ([https://arxiv.org/pdf/2105.08330.pdf](https://arxiv.org/pdf/2105.08330.pdf)).
+ Bags of Tricks (BoT)
  + GAT(norm.adj.)
  + Label Reuse
  + More details can be found in *Bag of Tricks for Node Classification with Graph Neural Networks* (https://arxiv.org/abs/2103.13355)
+ Self-Knowledge Distillation (self-KD)

#### Environmental Requirements

+ dgl >= 0.5.0
+ torch >= 1.6.0
+ torch_geometric >= 1.6.0
+ ogb == 1.3.0

#### Experiment Setup：

1. Generate node2vec embeddings, which save in `arxiv_embedding.pt`

   ```bash
   python node2vec_arxiv.py
   ```

2. Train the teacher model

   ```bash
   python3 mkteacher.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25
   ```

3. Train the student model

   ```bash
   python3 train_kd.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --alpha 0.9 --temp 0.7
   ```

#### Detailed Hyperparameter:

GAT:

```bash
Namespace(attn_drop=0.0, cpu=False, dropout=0.75, edge_drop=0.3, gpu=0, input_drop=0.25, log_every=20, lr=0.002, mask_rate=0.5, n_epochs=2000, n_heads=3, n_hidden=256, n_label_iters=1, n_layers=3, n_runs=10, no_attn_dst=True, plot_curves=False, save_pred=False, seed=0, use_embed=True, use_labels=True, use_norm=True, wd=0)

--n-runs N_RUNS         running times (default: 10)
--n-epochs N_EPOCHS     number of epochs (default: 2000)
--use-labels            Use labels in the training set as input features. (default: False)
--lr LR                 learning rate (default: 0.002)
--n-layers N_LAYERS     number of layers (default: 3)
--n-heads N_HEADS       number of heads (default: 3)
--n-hidden N_HIDDEN     number of hidden units (default: 256)
--dropout DROPOUT       dropout rate (default: 0.75)
--input-drop INPUT_DROP input drop rate (default: 0.25)
```

node2vec:

```bash
embedding_dim = 128
lr = 0.01
batch_size = 256
walk_length = 80
epochs = 5
```

#### Teacher Result:

```python
Namespace(attn_drop=0.0, cpu=False, dropout=0.75, edge_drop=0.3, gpu=0, input_drop=0.25, log_every=20, lr=0.002, mask_rate=0.5, n_epochs=2000, n_heads=3, n_hidden=256, n_label_iters=1, n_layers=3, n_runs=10, no_attn_dst=True, plot_curves=False, save_pred=False, seed=0, use_embed=True, use_labels=True, use_norm=True, wd=0)
Runned 10 times
Val Accs: [0.7463002114164905, 0.746568676801235, 0.7479110037249572, 0.7471056075707239, 0.7474747474747475, 0.7470049330514447, 0.7499916104567267, 0.7491526561294003, 0.7501929594952851, 0.7505620993993087]
Test Accs: [0.7402012221467811, 0.740180647285147, 0.740180647285147, 0.7403863959014876, 0.7407155936876324, 0.7410447914737773, 0.7411476657819476, 0.740160072423513, 0.74055099479456, 0.7407155936876324]
Average val accuracy: 0.748226450552032 ± 0.0015220916019935242
Average test accuracy: 0.7405283624467625 ± 0.0003500690422625202
Number of params: 1700432
```

#### Student Result:

```python
Namespace(alpha=0.9, attn_drop=0.0, cpu=False, dropout=0.75, edge_drop=0.3, gpu=0, input_drop=0.25, log_every=20, lr=0.002, mask_rate=0.5, n_epochs=1080, n_heads=3, n_hidden=256, n_label_iters=1, n_layers=3, n_runs=10, no_attn_dst=True, plot_curves=False, save_pred=False, seed=0, temp=0.7, use_embed=True, use_labels=True, use_norm=True, wd=0)
Runned 10 times
Val Accs: [0.7453941407429779, 0.7486492835330044, 0.7483472599751669, 0.7485821671868184, 0.7483137018020739, 0.7484814926675392, 0.7501929594952851, 0.7506292157454948, 0.7477096546863988, 0.7461324205510252]
Test Accs: [0.7424644569265272, 0.7421146842787483, 0.7420735345554801, 0.7421558340020163, 0.742032384832212, 0.7422381334485526, 0.7413534143982882, 0.7412711149517519, 0.7421764088636504, 0.7420118099705779]
Average val accuracy: 0.7482432296385785 ± 0.001506791867504243
Average test accuracy: 0.7419891776227805 ± 0.0003599660097975198
Number of params: 1700432
```

| Model                        | Test Accuracy   | Valid Accuracy  | Parameters | Hardware          |
| ---------------------------- | --------------- | --------------- | ---------- | ----------------- |
| GAT-node2vec + BoT           | 0.7405 ± 0.0004 | 0.7482 ± 0.0015 | 1700432    | Tesla V100 (32GB) |
| GAT-node2vec + BoT + self-KD | 0.7420 ± 0.0004 | 0.7482 ± 0.0015 | 1700432    | Tesla V100 (32GB) |

