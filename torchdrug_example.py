import torch
from torchdrug import datasets
from models_zinc import *

import torch
from torchdrug import data, datasets
from torchdrug import core, models, tasks, utils

dataset = datasets.ClinTox("~/molecule-datasets/")
lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = torch.utils.data.random_split(dataset, lengths)

graphs = []
labels = []
for i in range(4):
    sample = dataset[i]
    graphs.append(sample.pop("graph"))
    label = ["%s: %d" % (k, v) for k, v in sample.items()]
    label = ", ".join(label)
    labels.append(label)
graph = data.Molecule.pack(graphs)
graph.visualize(labels, num_row=1)


model= MMAConv(input_dim=dataset.node_feature_dim, hidden_dims=[256, 256, 256, 256], edge_input_dim=None, short_cut=True, batch_norm=True,
                 activation="relu", concat_hidden=True)

#model = models.RGCN(input_dim=dataset.node_feature_dim,
#                    num_relation=dataset.num_bond_type,
#                    hidden_dims=[256, 256, 256, 256], batch_norm=False)
task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                criterion="bce", metric=("auprc", "auroc"))

optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                     gpus=[0], batch_size=1024)
solver.train(num_epoch=100)
solver.evaluate("valid")
