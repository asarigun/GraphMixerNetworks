import torch
from torchdrug import datasets
from models_zinc import MMAConv
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from torch_geometric.datasets import ZINC
#from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool

dataset = datasets.ZINC250k("~/molecule-datasets/", kekulize=True,
                            node_feature="symbol")
# with open("path_to_dump/zinc250k.pkl", "wb") as fout:
#     pickle.dump(dataset, fout)
# with open("path_to_dump/zinc250k.pkl", "rb") as fin:
#     dataset = pickle.load(fin)
print("input_dim:", dataset.node_feature_dim)
print("input_dim:", type(dataset.node_feature_dim))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MMAConv(input_dim=dataset.node_feature_dim, hidden_dims=[256, 256, 256, 256], edge_input_dim=None, short_cut=False, batch_norm=False,
                 activation="relu", concat_hidden=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()

    total_error = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        total_error += (out.squeeze() - data.y).abs().sum().item()
    return total_error / len(loader.dataset)


for epoch in range(1, 301):
    loss = train(epoch)
    val_mae = test(val_loader)
    test_mae = test(test_loader)
    scheduler.step(val_mae)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_mae:.4f}, '
          f'Test: {test_mae:.4f}')



