'''
Author: Xiaoxiao Li
Date: 2019/02/24
'''

import os.path as osp
from os import listdir
import glob
import h5py

import torch
import numpy as np
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial

def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess



def read_data(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    y_list = []
    pseudo = []
    edge_att_list, edge_index_list,att_list = [], [], []

    #res = read_sigle_data(data_dir, onlyfiles[0])

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    func = partial(read_sigle_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        pseudo.append(np.diag(np.ones(res[j][4])))

    edge_att_arr = np.concatenate(edge_att_list)
    edge_index_arr = np.concatenate(edge_index_list, axis=1)
    att_arr = np.concatenate(att_list, axis=0)
    y_arr = np.stack(y_list)
    pseudo_arr = np.concatenate(pseudo, axis=0)
    #edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    edge_att_torch = torch.from_numpy(edge_att_arr).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).long()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    #data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch)
    pseudo_torch = torch.from_numpy(pseudo_arr).float()
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos = pseudo_torch )
    data, slices = split(data, batch_torch)

    return data, slices


def read_sigle_data(data_dir,filename):

    temp = h5py.File(osp.join(data_dir, filename), 'r')

    # read edge and edge attribute
    pcorr = np.abs(temp['pcorr'].value)
    # only keep the top 10% edges
    th = np.percentile(pcorr.reshape(-1),95)
    pcorr[pcorr < th] = 0  # set a threshold
    num_nodes = pcorr.shape[0]

    G = from_numpy_matrix(pcorr)
    A = nx.to_scipy_sparse_matrix(G)
    adj = A.tocoo()
    edge_att = np.zeros((len(adj.row)))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]
    edge_index = np.stack([adj.row, adj.col])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index).long(), torch.from_numpy(edge_att).float())
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes,
                                    num_nodes)

    att = temp['corr'].value

    return edge_att.data.numpy(),edge_index.data.numpy(),att,temp['indicator'].value, num_nodes


