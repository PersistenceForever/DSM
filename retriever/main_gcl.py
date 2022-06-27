import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
import random

from util import load_data
from models.clf_model import Classifier
from models.dgi import DGI
import sys
import os
import pickle

sig = torch.nn.Sigmoid()

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  

def preprocess_neighbors_sumavepool(edge_index, nb_nodes, device):
    adj_idx = edge_index
        
    # adj_idx_2 = torch.cat([torch.unsqueeze(adj_idx[1], 0), torch.unsqueeze(adj_idx[0], 0)], 0)
    # adj_idx = torch.cat([adj_idx, adj_idx_2], 1)

    self_loop_edge = torch.LongTensor([range(nb_nodes), range(nb_nodes)])
    adj_idx = torch.cat([adj_idx, self_loop_edge], 1)
        
    adj_elem = torch.ones(adj_idx.shape[1])

    adj = torch.sparse.FloatTensor(adj_idx, adj_elem, torch.Size([nb_nodes, nb_nodes]))

    return adj.to(device)

def evaluate(model, test_graph):
    output = model(test_graph[0], test_graph[1])
    pred = sig(output.detach().cpu())
    test_idx = test_graph[3]
    
    labels = test_graph[-1]
    pred = pred[labels[test_idx, 0].astype('int')].numpy()
    target = labels[test_idx, 1]
    
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(target, pred, pos_label=1)
    auc = metrics.auc(false_positive_rate, true_positive_rate)

    return auc

def finetune(args, model_pretrain, device, test_graph, feats_num):
    # initialize the joint model
    model = Classifier(args.num_layers, args.num_mlp_layers, feats_num, args.hidden_dim, args.final_dropout, args.neighbor_pooling_type, device).to(device)
    
    # replace the encoder in joint model with the pre-trained encoder
    pretrained_dict = model_pretrain.state_dict()
    model_dict = model.state_dict()
    pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    criterion_tune = nn.BCEWithLogitsLoss()

    res = []
    train_idx = test_graph[2]
    node_train = test_graph[-1][train_idx, 0].astype('int')
    label_train = torch.FloatTensor(test_graph[-1][train_idx, 1]).to(device)
    for _ in range(1, args.finetune_epochs+1):
        model.train()
        output = model(test_graph[0], test_graph[1])
        loss = criterion_tune(output[node_train], torch.reshape(label_train, (-1, 1)))
        
        #backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # testing
        model.eval()
        auc = evaluate(model, test_graph)
        res.append(auc)

    return np.max(res)


def main():
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net')
    parser.add_argument('--dataset', type=str, default="WQTrainGraph",
                        help='name of dataset (default: WQTrainGraph_0/WQTestGraph_0)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--finetune_epochs', type=int, default=100,
                        help='number of finetune epochs (default: 100)')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='number of layers (default: 2)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=1024,
                        help='number of hidden units (default: 1024)')
    parser.add_argument('--n_views', type=int, default=2,
                        help='number of views (default: k + 1)')
    parser.add_argument('--batchsize', type=int, default=16,
                        help='size of batch (default: 256)')
    parser.add_argument('--num_folds', type=int, default=10,
                        help='number of folds (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over neighboring nodes: sum or average')
    args = parser.parse_args()

    setup_seed(0)
    
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    # Data loading
    edge_index, feats, nb_nodes = load_data(args.dataset)
    input_dim = feats.shape[1]
    # the shuffled features are used to contruct the negative samples
    idx = np.random.permutation(nb_nodes)
    shuf_feats = feats[idx, :]
   
    model_pretrain = DGI(args.num_layers, args.num_mlp_layers, input_dim, args.hidden_dim, args.neighbor_pooling_type, device).to(device)
   
    optimizer_train = optim.Adam(model_pretrain.parameters(), lr=args.lr)

  

    batch_size = 1
    lbl_1 = torch.ones(batch_size, nb_nodes)
    lbl_2 = torch.zeros(batch_size, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

    adj = preprocess_neighbors_sumavepool(torch.LongTensor(edge_index), nb_nodes, device)
    feats = torch.FloatTensor(feats).to(device)
    shuf_feats = torch.FloatTensor(shuf_feats).to(device)

    with open('./train_StructureSimilar_WQ_Path_1.pt', 'rb') as f:
        pathDict = pickle.load(f) # pathDict[index][0] is the index's top-20 the most similar subgraph's id. it is a list
    train_nodeSet = np.load('./wholeGraphData/WQTrainGraphNodeSet.npy', allow_pickle=True)
    cnt_wait = 0
    best = 1e9
    best_t = 1
    patience = 15

    # calculate the test graph'sb node embedding
    # adj = preprocess_neighbors_sumavepool(torch.LongTensor(edge_index), nb_nodes, device)
    # model_pretrain.load_state_dict(torch.load('./best_qa_param_20_001.pt'))
    # emb = model_pretrain.get_emb(feats, adj)
    # np.save('./QA_Process/' + args.dataset +'_node1024.npy', emb)
    # print("calculate success!")
    # sys.exit(0)

    # pre-training
    model_pretrain.train()
    for epoch in range(1, args.epochs + 1):
        loss_pretrain = model_pretrain(feats, adj, train_nodeSet, pathDict, args.batchsize)
               
        print("loss:", loss_pretrain.item())
        with open('./trainLoss.txt', 'a+') as ft:
            ft.write(str(loss_pretrain.item()))
            ft.write('\n')

        # if epoch > 10:
        if loss_pretrain.item() < best:
            best = loss_pretrain.item()
            best_t = epoch
            cnt_wait = 0
            torch.save(model_pretrain.state_dict(), './best_WQ_param_1_0001.pt', _use_new_zipfile_serialization = False) 
        else:
            cnt_wait +=1
        
        if cnt_wait == patience:
            print("Early Stopping!, epoch:", epoch)
            break

        if optimizer_train is not None:
            optimizer_train.zero_grad()
            loss_pretrain.backward()         
            optimizer_train.step()

    #load model 
    model_pretrain.load_state_dict(torch.load('./best_WQ_param_1_0001.pt')) 

    emb = model_pretrain.get_emb(feats, adj)
    np.save('./CLGraphData/' + args.dataset +'_node1024_1_16_0001.npy', emb)
    # np.save('./wholeGraphData/' + args.dataset +'_node1024.npy', emb)
    print('Pre-training done!')

    #fine-tuning process
    # fold_idx = 1
    # every_fold_auc = []
    # for (train_idx, test_idx) in split_idx:
    #     test_graph = (feats, adj, train_idx, test_idx, label)
    #     tmp_auc = finetune(args, model_pretrain, device, test_graph, input_dim)
    #     every_fold_auc.append(tmp_auc)
    #     print('AUC on the Fold'+str(fold_idx)+': ', tmp_auc)
    #     fold_idx += 1
    
    # print('The averaged AUC score: ', np.mean(every_fold_auc))

if __name__ == '__main__':
    main()