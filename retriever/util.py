import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

def load_data(datasets):
    # load the adjacency
    adj = np.loadtxt('./data/'+datasets+'.txt')
    # num_user = len(set(adj[:, 0]))
    # num_object = len(set(adj[:, 1]))
    adj = adj.astype('int')
    nb_nodes = np.max(adj) + 1
    edge_index = adj.T
    print('Load the edge_index done!')
    
    # load the user label
    # label = np.loadtxt('./data/'+datasets+'_label.txt')
    # y = label[:, 1]
    # print('Ratio of fraudsters: ', np.sum(y) / len(y))
    # print('Number of edges: ', edge_index.shape[1])
    # print('Number of users: ', num_user)
    # print('Number of objects: ', num_object)
    # print('Number of nodes: ', nb_nodes)

    # split the train_set and validation_set
    # split_idx = []
    # skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    # for (train_idx, test_idx) in skf.split(y, y):
    #     split_idx.append((train_idx, test_idx))
   
    # load initial features
    feats = np.load('./features/'+datasets+'_feature1024.npy')
    return edge_index, feats, nb_nodes


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.len = len(inputs)
        self.data = inputs
    def __getitem__(self, index):
              
        return self.data[index]
    def __len__(self):
        return self.len
