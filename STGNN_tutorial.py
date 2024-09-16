# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)


import tsl
import torch
import numpy as np
import pandas as pd

print(f"tsl version  : {tsl.__version__}")
print(f"torch version: {torch.__version__}")

pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(edgeitems=3, precision=3)
torch.set_printoptions(edgeitems=2, precision=3)

# Utility functions ################
def print_matrix(matrix):
    return pd.DataFrame(matrix)

def print_model_size(model):
    tot = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out = f"Number of model ({model.__class__.__name__}) parameters:{tot:10d}"
    print("=" * len(out))
    print(out)

from tsl.datasets import MetrLA

dataset = MetrLA(root='./data')
#
# print(dataset)

print(f"Sampling period: {dataset.freq}")
print(f"Has missing values: {dataset.has_mask}")
print(f"Percentage of missing values: {(1 - dataset.mask.mean()) * 100:.2f}%")
print(f"Has exogenous variables: {dataset.has_covariates}")
print(f"Covariates: {', '.join(dataset.covariates.keys())}")

print_matrix(dataset.dist)
dataset.dataframe()

print(f"Default similarity: {dataset.similarity_score}")
print(f"Available similarity options: {dataset.similarity_options}")
print("==========================================")

sim = dataset.get_similarity("distance")  # or dataset.compute_similarity()

print("Similarity matrix W:")
print_matrix(sim)

connectivity = dataset.get_connectivity(threshold=0.1,
                                        include_self=False,
                                        normalize_axis=1,
                                        layout="edge_index")

edge_index, edge_weight = connectivity

print(f'edge_index {edge_index.shape}:\n', edge_index)
print(f'edge_weight {edge_weight.shape}:\n', edge_weight)

from tsl.ops.connectivity import edge_index_to_adj

adj = edge_index_to_adj(edge_index, edge_weight)
print(f'A {adj.shape}:')
print_matrix(adj)

print(f'Sparse edge weights:\n', adj[edge_index[1], edge_index[0]])

from tsl.data import SpatioTemporalDataset

torch_dataset = SpatioTemporalDataset(target=dataset.dataframe(),
                                      connectivity=connectivity,
                                      mask=dataset.mask,
                                      horizon=12,
                                      window=12,
                                      stride=1)
print(torch_dataset)
sample = torch_dataset[0]
print(sample)

sample.input.to_dict()
sample.target.to_dict()

if sample.has_mask:
    print(sample.mask)
else:
    print("Sample has no mask.")

if sample.has_transform:
    print(sample.transform)
else:
    print("Sample has no transformation functions.")

print(sample.pattern)
print("==================   Or we can print patterns and shapes together   ==================")
print(sample)

batch = torch_dataset[:5]
print(batch)

from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter)
from tsl.data.preprocessing import StandardScaler

# Normalize data using mean and std computed over time and node dimensions
scalers = {'target': StandardScaler(axis=(0, 1))}

# Split data sequentially:
#   |------------ dataset -----------|
#   |--- train ---|- val -|-- test --|
splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

dm = SpatioTemporalDataModule(
    dataset=torch_dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=64,
)

print(dm)
dm.setup()
print(dm)

import torch.nn as nn

from tsl.nn.blocks.encoders import RNN
from tsl.nn.layers import NodeEmbedding, DiffConv
from einops.layers.torch import Rearrange  # reshape data with Einstein notation


class TimeThenSpaceModel(nn.Module):
    def __init__(self, input_size: int, n_nodes: int, horizon: int,
                 hidden_size: int = 32,
                 rnn_layers: int = 1,
                 gnn_kernel: int = 2):
        super(TimeThenSpaceModel, self).__init__()

        self.encoder = nn.Linear(input_size, hidden_size)

        self.node_embeddings = NodeEmbedding(n_nodes, hidden_size)

        self.time_nn = RNN(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=rnn_layers,
                           cell='gru',
                           return_only_last_state=True)

        self.space_nn = DiffConv(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 k=gnn_kernel)

        self.decoder = nn.Linear(hidden_size, input_size * horizon)
        self.rearrange = Rearrange('b n (t f) -> b t n f', t=horizon)

    def forward(self, x, edge_index, edge_weight):
        # x: [batch time nodes features]
        x_enc = self.encoder(x)  # linear encoder: x_enc = xΘ + b
        x_emb = x_enc + self.node_embeddings()  # add node-identifier embeddings
        h = self.time_nn(x_emb)  # temporal processing: x=[b t n f] -> h=[b n f]
        z = self.space_nn(h, edge_index, edge_weight)  # spatial processing
        x_out = self.decoder(z)  # linear decoder: z=[b n f] -> x_out=[b n t⋅f]
        x_horizon = self.rearrange(x_out)
        return x_horizon

hidden_size = 32   #@param
rnn_layers = 1     #@param
gnn_kernel = 2     #@param

input_size = torch_dataset.n_channels   # 1 channel
n_nodes = torch_dataset.n_nodes         # 207 nodes
horizon = torch_dataset.horizon         # 12 time steps

stgnn = TimeThenSpaceModel(input_size=input_size,
                           n_nodes=n_nodes,
                           horizon=horizon,
                           hidden_size=hidden_size,
                           rnn_layers=rnn_layers,
                           gnn_kernel=gnn_kernel)
print(stgnn)
print_model_size(stgnn)

from tsl.metrics.torch import MaskedMAE, MaskedMAPE
from tsl.engines import Predictor

loss_fn = MaskedMAE()

metrics = {'mae': MaskedMAE(),
           'mape': MaskedMAPE(),
           'mae_at_15': MaskedMAE(at=2),  # '2' indicates the third time step,
                                          # which correspond to 15 minutes ahead
           'mae_at_30': MaskedMAE(at=5),
           'mae_at_60': MaskedMAE(at=11)}

# setup predictor
predictor = Predictor(
    model=stgnn,                   # our initialized model
    optim_class=torch.optim.Adam,  # specify optimizer to be used...
    optim_kwargs={'lr': 0.001},    # ...and parameters for its initialization
    loss_fn=loss_fn,               # which loss function to be used
    metrics=metrics                # metrics to be logged during train/val/test
)

from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger(save_dir="logs", name="tsl_intro", version=0)

# %load_ext tensorboard
# %tensorboard --logdir logs

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)

# trainer = pl.Trainer(max_epochs=100,
#                      logger=logger,
#                      gpus=1 if torch.cuda.is_available() else None,
#                      limit_train_batches=100,  # end an epoch after 100 updates
#                      callbacks=[checkpoint_callback])
#
# trainer.fit(predictor, datamodule=dm)
import pytorch_lightning as pl
import torch

# GPUが利用可能かどうかをチェックして、デバイスの設定を行う
if torch.cuda.is_available():
    devices = 1
    accelerator = 'gpu'
else:
    devices = 1  # CPUでも1つのデバイス（コア）を指定
    accelerator = 'cpu'

# Trainerの設定
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    devices=devices,                  # 使用するデバイスの数
    accelerator=accelerator,          # 'gpu'または'cpu'を指定
    limit_train_batches=100,          # エポックを100バッチで終了
    callbacks=[checkpoint_callback]   # コールバックの設定
)


# モデルの学習
trainer.fit(predictor, datamodule=dm)


################

predictor.load_model(checkpoint_callback.best_model_path)
predictor.freeze()

trainer.test(predictor, datamodule=dm);