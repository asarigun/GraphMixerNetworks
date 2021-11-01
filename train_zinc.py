import sys
import os
import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dgl import DGLGraph
import dgl
import pickle
import csv
from sklearn.metrics import f1_score
from ogb.nodeproppred import Evaluator

from zinc import *
from scipy import sparse as sp
#from dgl.data import register_data_args, load_data
from models_zinc import *
from utils_zinc import *
#from dataset.visualization import vis_graph
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="MMAConv", help="MMAConv")
parser.add_argument("--dataset", type=str, default="zinc", help="ZINC")
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")

parser.add_argument("--log-subdir", type=str, default="run0000", help="the subdir name of log, eg. run0000")
parser.add_argument("--ncaps", type=int, default=8,help="")
parser.add_argument("--nhidden", type=int, default=16, help="")
parser.add_argument("--routit", type=int, default=5, help="")
parser.add_argument("--nlayer", type=int, default=4, help="")
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument("--in_dim", type=int, default=36, help="dim of embedded feature")
parser.add_argument("--num_latent", type=int, default=8, help="number of training epochs")
parser.add_argument("--num_hidden", type=int, default=18*8, help="number of hidden units")
parser.add_argument("--dis_weight", type=float, default=0.5, help="weight of disentangle")
parser.add_argument("--in_drop", type=float, default=0.0, help="input feature dropout")
parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0, help="weight decay")
parser.add_argument('--negative_slope', type=float, default=0.2, help="the negative slope of leaky relu")
parser.add_argument('--early_stop', action='store_true', default=False, help="indicates whether to use early stop or not")
parser.add_argument('--seed', type=int, default=42, help="set seed")
parser.add_argument('--aggregators', type=str, default="mean,max,min", help='choose your aggregators')

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev)  
print("Device:",device)

args = parser.parse_args()



def test(model, data_loader):
    loss_fcn = torch.nn.L1Loss()
    
    model.eval()
    loss = 0
    mae = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].cuda()  # num x feat
            batch_e = batch_graphs.edata['feat'].cuda()
            batch_snorm_e = batch_snorm_e.cuda()
            batch_targets = batch_targets.cuda()
            batch_snorm_n = batch_snorm_n.cuda()         # num x 1
            
            model.g = batch_graphs
            batch_scores = model.forward(batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            
            loss = loss_fcn(batch_scores, batch_targets)
            iter_loss = loss.item()
            iter_mae = F.l1_loss(batch_scores, batch_targets).item()
            loss += iter_loss
            mae += iter_mae
        
    loss /= (iter + 1)
    mae /= (iter + 1)
    return loss, mae

""""
data_dir = './data/molecules'
zinc_data_train = MoleculeDGL('train', data_dir, num_graphs = 10000)
print("len of dataset:", len(zinc_data_train))
zinc_data_val = MoleculeDGL('val', data_dir, num_graphs = 1000)
print("len of dataset:", len(zinc_data_val))
zinc_data_test = MoleculeDGL('test', data_dir, num_graphs = 1000)
print("len of dataset:", len(zinc_data_test))
"""
zinc_data = MoleculeDatasetDGL()
#zinc_data.val.collate
print(zinc_data)


def main(args):
    #torch.cuda.set_device(args.gpu)
    set_seed(args)
    #global g,label,add_all
    log_dir = make_log_dir(args.model_name, args.dataset, args.log_subdir)
    
    log_file = os.path.join(log_dir, "log.txt")
    sys.stdout = open(log_file, 'w')
    backup_model = f"cp -r ./models {log_dir}"
    os.system(backup_model)
    
    data_dir = './data/molecules'
    # load and preprocess dataset
    #zinc_data = load_dataset()
    zinc_data = MoleculeDatasetDGL()
    #zinc_data_val = MoleculeDatasetDGL('val', data_dir, num_graphs = 1000)
    #zinc_data_test = MoleculeDatasetDGL('test', data_dir, num_graphs = 1000)

    print(len(zinc_data.train))
    #print(len(zinc_data_val))
    #print(len(zinc_data_test))
    #zinc_data = MoleculeDatasetDGL()
    """
    node_set = set()
    edge_set = set()
    for data in zinc_data.train:
        g, label, add_all = data
        print("graph_lists:",g)
        print("graph_labels:",label)
        print("add_all:",add_all)
        node_set = node_set.union( set(g.ndata['feat'].numpy()) )
        edge_set = edge_set.union( set(g.edata['feat'].numpy()) )
    print(node_set)
    print(edge_set)    
    """
    #print("len of dataset:", len(zinc_data.train)) 
    
    train_loader = DataLoader(zinc_data.train, batch_size=1000, shuffle=True, collate_fn=zinc_data.train.collate, num_workers=4)
    #val_loader = DataLoader(zinc_data_val, batch_size=1000, shuffle=False, collate_fn=zinc_data_val.collate)
    #test_loader = DataLoader(zinc_data_test, batch_size=1000, shuffle=False, collate_fn=zinc_data_test.collate)
    
    #graph_lists[idx], graph_labels[idx], add_all[idx] = zinc_data_val.__getitem__(idx)

    #g = graph_lists[idx]
    #label = graph_labels[idx]
    #add_all = add_all[idx]
    # placeholder of dataset
    #g, features, labels, train_mask, val_mask, test_mask, factor_graphs = dataset
    ## dataset = (None, None, None, None, None, None, None)
    # create model
    #graph_lists, graph_labels, add_all = MoleculeDGL(split='train', data_dir='data/molecules', num_graphs=None)
    #graph_lists, graph_labels, add_all = MoleculeDGL(split='val', data_dir='data/molecules', num_graphs=None)
    #g, label, add_all = MoleculeDGL(split='test', data_dir='data/molecules', num_graphs=None)
    #g = graph_lists

    #zinc_dataset = MoleculeDatasetDGL()
    #(g, label, add_all) = load_dataset()
    #node_set = set()
    #edge_set = set()
    #for data in zinc_dataset.train:
    #    g, label, add_all = data
    #    node_set = node_set.union( set(g.ndata['feat'].numpy()) )
    #    edge_set = edge_set.union( set(g.edata['feat'].numpy()) )
     
    model = MMAConv(#add_all, 
                        g, 
                        nfeat = args.in_dim, 
                        nhid = args.num_hidden, 
                        num_atom_type = 28, 
                        num_bond_type = 4, 
                        dropout = args.dropout, 
                        aggregator_list = args.aggregators.split(","), 
                        device = device)

    print(model)
    # define loss func
    loss_fcn = torch.nn.L1Loss()
    
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr, 
                                weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5,
                                                     patience=50,
                                                     verbose=True)

    best_val_loss = sys.maxsize
    best_test_mae = sys.maxsize
    dur = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_train_mae = 0
        t0 = time.time()
        for iter, (batch_graphs, batch_targets, batch_snorm_n, batch_snorm_e) in enumerate(train_loader):
            batch_x = batch_graphs.ndata['feat']  # num x feat
            batch_e = batch_graphs.edata['feat']
            batch_snorm_e = batch_snorm_e
            batch_targets = batch_targets
            batch_snorm_n = batch_snorm_n         # num x 1
            
            optimizer.zero_grad()
            
            model.g = batch_graphs
            batch_scores = model.forward(batch_x, batch_e, batch_snorm_n, batch_snorm_e)
            
            loss = loss_fcn(batch_scores, batch_targets)
            
            #if args.model_name == "FactorGNN" and args.dis_weight > 0.0:
            #                                        losses = model.compute_disentangle_loss()
            #                                        dis_loss = model.merge_loss(losses) * args.dis_weight
            #                                        loss = loss + dis_loss

            loss.backward()
            optimizer.step()
            
            iter_loss = loss.item()
            iter_mae = F.l1_loss(batch_scores, batch_targets).item()
            epoch_loss += iter_loss
            epoch_train_mae += iter_mae
        
        dur.append(time.time() - t0)
        epoch_loss /= (iter + 1)
        epoch_train_mae /= (iter + 1)
        # print(f"loss {epoch_loss:.4f}, mae {epoch_train_mae:.4f}")
        val_loss, val_mae = test(model, val_loader)
        test_loss, test_mae = test(model, test_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_mae = test_mae
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_test_mae,
                        'args': args}, 
                        os.path.join(log_dir, 'best_model.pt') )

        print(  f"time {np.mean(dur):.2f} epoch {epoch:03d} | " + 
                f"train ({epoch_loss:.4f}, {epoch_train_mae:.4f}) | "+
                f"val ({val_loss:.4f}, {val_mae:.4f}) | "+
                f"test ({test_loss:.4f}, {test_mae:.4f}) | "+
                f"best: {best_test_mae:.4f}")
        
        sys.stdout.flush()
        
        if optimizer.param_groups[0]['lr'] > 1e-5:
            scheduler.step(val_loss)

main(args)        
