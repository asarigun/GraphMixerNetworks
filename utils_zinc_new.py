# code are partially from https://github.com/graphdeeplearning/benchmarking-gnns
import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np
import csv
import dgl
import networkx as nx

from sklearn.metrics import f1_score
from ogb.nodeproppred import Evaluator

from scipy import sparse as sp
import numpy as np

# *NOTE
# The dataset pickle and index files are in ./zinc_molecules/ dir
# [<split>.pickle and <split>.index; for split 'train', 'val' and 'test']
cur_dir = os.path.dirname(os.path.abspath(__file__))
zinc_dir = os.path.join(cur_dir, "data/molecules")


def MoleculeDGL(split, data_dir, num_graphs=None):    

    file_path = os.path.join(cur_dir, 'data/molecules', f'graph_list_labels_{split}.pt')
    if not os.path.isfile(file_path):
        with open(data_dir + "/%s.pickle" % split,"rb") as f:
            data = pickle.load(f)
    
        # loading the sampled indices from file ./zinc_molecules/<split>.index
        with open(data_dir + "/%s.index" % split,"r") as f:
            data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
            data = [ data[i] for i in data_idx[0] ]
        print("len(data):", len(data))
        print("num_graphs:", num_graphs)
        assert len(data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k" 
                      
    """
    with open(data_dir + "/%s.pickle" % split,"rb") as f:
        data = pickle.load(f)
    
    if num_graphs in [10000, 1000]:
        # loading the sampled indices from file ./zinc_molecules/<split>.index
        with open(data_dir + "/%s.index" % split,"r") as f:
            data_idx = [list(map(int, idx)) for idx in csv.reader(f)]
            data = [ data[i] for i in data_idx[0] ]

        assert len(data)==num_graphs, "Sample num_graphs again; available idx: train/val/test => 10k/1k/1k"
    """
    """      
    data is a list of Molecule dict objects with following attributes
        
      molecule = data[idx]
    ; molecule['num_atom'] : nb of atoms, an integer (N)
    ; molecule['atom_type'] : tensor of size N, each element is an atom type, an integer between 0 and num_atom_type
    ; molecule['bond_type'] : tensor of size N x N, each element is a bond type, an integer between 0 and num_bond_type
    ; molecule['logP_SA_cycle_normalized'] : the chemical property to regress, a float variable
    """
 
 
    graph_lists = []
    graph_labels = []
    add_all = []
    n_samples = len(graph_lists)
    

    if os.path.isfile(file_path):
        print(f"load from {file_path}")
        with open(file_path, 'rb') as f:
            graph_lists, graph_labels, add_all = pickle.load( f )

    print("preparing %d graphs for the %s set..." % (num_graphs, split.upper()))
    
    for molecule in data:
        node_features = molecule['atom_type'].long()
            
        adj = molecule['bond_type']
        #print("adj:",adj.shape) 
        #adj_type = nx.adjacency_matrix(nx.from_dict_of_lists(molecule['bond_type']))
        for i in range(adj.shape[0]):
            add_all.append(adj[i].nonzero())

        edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list
            
        edge_idxs_in_adj = edge_list.split(1, dim=1)
        edge_features = adj[edge_idxs_in_adj].reshape(-1).long()
            
        # Create the DGL Graph
        g = dgl.DGLGraph()
        g.add_nodes(molecule['num_atom'])
        g.ndata['feat'] = node_features
            
        for src, dst in edge_list:
            g.add_edges(src.item(), dst.item())
        g.edata['feat'] = edge_features
            
        graph_lists.append(g)
        graph_labels.append(molecule['logP_SA_cycle_normalized'])
        #for lbl in molecule['logP_SA_cycle_normalized']:
        #    lbl = np.where(lbl==1)[0]
        #    graph_labels.append(lbl[0] if list(lbl)!=[] else 0)
        
    with open(file_path, 'wb') as f:
        pickle.dump((graph_lists, graph_labels, add_all), f)

    print("graph_lists:",len(graph_lists))
    print("graph_labels:",len(graph_labels))
    print("add_all:",len(add_all))
    return graph_lists, graph_labels, add_all


class MoleculeDatasetDGL(torch.utils.data.Dataset):
    def __init__(self, name='Zinc'):
        t0 = time.time()
        self.name = name
        
        self.num_atom_type = 28 # known meta-info about the zinc dataset; can be calculated as well
        self.num_bond_type = 4 # known meta-info about the zinc dataset; can be calculated as well
        
        data_dir='./data/molecules'
        if self.name == 'ZINC-full':
            data_dir='./data/molecules/zinc_full'
            self.train = MoleculeDGL('train', data_dir, num_graphs=220011)
            self.val = MoleculeDGL('val', data_dir, num_graphs=24445)
            self.test = MoleculeDGL('test', data_dir, num_graphs=5000)
        else:            
            self.train = MoleculeDGL('train', data_dir, num_graphs=10000)
            self.val = MoleculeDGL('val', data_dir, num_graphs=1000)
            self.test = MoleculeDGL('test', data_dir, num_graphs=1000)
        print("Time taken: {:.4f}s".format(time.time()-t0))
    
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels, add_all = map(list, zip(*samples))
        print("add_all in collate:", add_all)
        # graphs = [self_loop(g) for g in graphs]
        labels = torch.from_numpy(np.array(labels))
        labels = torch.tensor(labels).unsqueeze(1)
        #labels = torch.tensor(np.array(labels)).unsqueeze(1)
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels, snorm_n, snorm_e

def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in MoleculeDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g



def positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVec = EigVec[:, EigVal.argsort()] # increasing order
    # g.ndata['pos_enc'] = torch.from_numpy(np.abs(EigVec[:,1:pos_enc_dim+1])).float() 
    
    return g



class MoleculeDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading SBM datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        data_dir = 'data/molecules'
        with open(data_dir+name+'.pkl',"rb") as f:
            f = pickle.load(f)
            self.train = f[0]
            self.val = f[1]
            self.test = f[2]
            self.num_atom_type = f[3]
            self.num_bond_type = f[4]
        print('train, test, val sizes :',len(self.train),len(self.test),len(self.val))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = torch.cat(tab_snorm_n).sqrt()  
        #tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        #tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        #snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples, edge_feat):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels)).unsqueeze(1)
        #tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        #tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        #snorm_n = tab_snorm_n[0][0].sqrt()  
        
        #batched_graph = dgl.batch(graphs)
    
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)

        if edge_feat:
            # use edge feats also to prepare adj
            adj_with_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type + self.num_bond_type)])
            adj_with_edge_feat = torch.cat([adj.unsqueeze(0), adj_with_edge_feat], dim=0)

            us, vs = g.edges()      
            for idx, edge_label in enumerate(g.edata['feat']):
                adj_with_edge_feat[edge_label.item()+1+self.num_atom_type][us[idx]][vs[idx]] = 1

            for node, node_label in enumerate(g.ndata['feat']):
                adj_with_edge_feat[node_label.item()+1][node][node] = 1
            
            x_with_edge_feat = adj_with_edge_feat.unsqueeze(0)
            
            return None, x_with_edge_feat, labels
        
        else:
            # use only node feats to prepare adj
            adj_no_edge_feat = torch.stack([zero_adj for j in range(self.num_atom_type)])
            adj_no_edge_feat = torch.cat([adj.unsqueeze(0), adj_no_edge_feat], dim=0)

            for node, node_label in enumerate(g.ndata['feat']):
                adj_no_edge_feat[node_label.item()+1][node][node] = 1

            x_no_edge_feat = adj_no_edge_feat.unsqueeze(0)

            return x_no_edge_feat, None, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim = 0)#.squeeze()
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))
    
    def _add_self_loops(self):
        
        # function for adding self loops
        # this function will be called only if self_loop flag is True
            
        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]


def load_dataset():
    # if args is not provide, use dataset
    zinc_data = MoleculeDatasetDGL()
    return zinc_data

evaluator = Evaluator(name = "ogbn-proteins")
# print(evaluator.expected_input_format) 
# print(evaluator.expected_output_format) 

def evaluate_multilabel(model, feature, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = torch.sigmoid(model(features))
        logits = logits[mask]
        labels = labels[mask]
        return evaluator({'y_true': labels, 'y_pred': logits})


def evaluate_f1(logits, labels):
    # logits = torch.sigmoid(logits)
    y_pred = torch.where(logits > 0.0, torch.ones_like(logits), torch.zeros_like(logits))
    y_pred = y_pred.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average='micro')


def accuracy(logits, labels):
    
    
    _, indices = torch.max(logits, dim=1)
    if len(indices.shape) > 1:
        indices = indices.view(-1,)
        labels = labels.view(-1,)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

loss_fcn = torch.nn.CrossEntropyLoss()
def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        if len(labels.shape) > 1:
            logits = logits[mask].view(-1, torch.max(labels) + 1, labels.shape[1])
        else:
            logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels), loss_fcn(logits, labels).item()


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score <= self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')
    
    def load_checkpoint(self, model):
        model.load_state_dict( torch.load('es_checkpoint.pt') )
        model.eval()


def set_seed(args=None):
    seed = 1 if not args else args.seed
    
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)


def make_log_dir(model_name, dataset, subdir):
    # make and return
    model_name = model_name.lower()
    log_dir = os.path.join(f"./data/run_log/{dataset}", model_name, subdir)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

"""
if __name__ == '__main__':
    
    zinc_dataset = MoleculeDatasetDGL()
    node_set = set()
    edge_set = set()
    for data in zinc_dataset.train:
        g, label, add_all = data
        node_set = node_set.union( set(g.ndata['feat'].numpy()) )
        edge_set = edge_set.union( set(g.edata['feat'].numpy()) )
    print(node_set)
    print(edge_set)
"""
if __name__ == '__main__':
    data_dir = 'data/molecules/'
    zinc_dataset = MoleculeDGL('train', data_dir, num_graphs=10000)
    node_set = set()
    edge_set = set()
    for data in zinc_dataset:
        g, label, add_all = data
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        #print("graph_lists:",g)
        #print("len graph_lists:", len(g)) 
        #print("graph_labels:",label)
        #print("len _graph_labels:", label.shape()) 
        #print("add_all:",add_all)
        #print("len add_all:", add_all.shape())
        node_set = node_set.union( set(g.ndata['feat'].numpy()) )
        edge_set = edge_set.union( set(g.edata['feat'].numpy()) )
    print("node_set:",node_set)
    print("edge_set:",edge_set)
