from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
import numpy as np
import h5py
import deepdish as dd
import pandas as pd
from sklearn.model_selection import KFold

def boxcox_plot(data):
    x = data.x.numpy()
    for i in range(9):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        prob = stats.probplot(x[:,i], dist=stats.norm, plot=ax1)
        ax1.set_xlabel('')
        ax1.set_title('Probplot against normal distribution')
        # ax2 = fig.add_subplot(212)
        # xt, lamb = stats.boxcox(x[:,i] - np.amin(x[:,i])+1)
        # prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
        # ax2.set_title('Probplot after Box-Cox transformation')
        # plt.show()


def boxcox_transform_train(x):
    xt, lamb = stats.boxcox(x - torch.min(x) + 1)
    #lamb = 0
    #xt_torch = xt
    xt_torch = torch.from_numpy(xt).float()
    xt_mean = torch.mean(xt_torch).float()
    xt_std = torch.std(xt_torch).float()
    xt_norm = (xt_torch-xt_mean)/xt_std
    return xt_norm,lamb,xt_mean, xt_std


def boxcox_transform_test(x,lamb, xt_mean, xt_std):
    if lamb == 0:
        y = torch.log(x)
    elif lamb!=0:
        y = ((x-torch.min(x)+1)**lamb-1)/lamb
    else:
        print("lambda is negative!")
        raise ValueError
    res = (y-xt_mean)/xt_std
    return res


def normal_transform_train(x):
    #xt, lamb = stats.boxcox(x - torch.min(x) + 1)
    lamb = 0
    #xt_torch = xt
    xt_mean = torch.mean(x).float()
    xt_std = torch.std(x).float()
    xt_norm = (x-xt_mean)/xt_std
    return xt_norm,lamb,xt_mean, xt_std


def normal_transform_test(x,lamb, xt_mean, xt_std):
    res = (x-xt_mean)/xt_std
    return res

def count_parameters(model):
    for p in model.parameters():
        if p.requires_grad:
            print('Number of parameter', p.numel())
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_test_split_noaug(mat_dir, kfold = 10, fold = 0):
    subjects = loadmat(mat_dir)
    pat_id = list(subjects['subjects_pat'])
    con_id = list(subjects['subjects_con'])
    sub_list = pat_id + con_id
    label = list(np.ones(len(pat_id)))+list(np.zeros(len(con_id)))
    list1, list2 = (list(t) for t in zip(*sorted(zip(sub_list, label))))


    x_ind = range(len(sub_list))
    y_ind = np.array(list2)

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=7)
    test_index = list()
    train_index = list()
    for a, b in skf.split(x_ind, y_ind):
        test_index.append(b)
        train_index.append(a)

    train_id = train_index[fold]
    test_id = test_index[fold]

    tr_index = list()
    te_index = list()
    count = 0
    for i in range(len(sub_list)):
        if list2[i] ==1:
            rep = 1
        else:
            rep = 1
        ll = list(range(count,count+rep))
        if i in train_id:
            tr_index.append(ll)
        else:
            te_index.append(ll)
        count = count + rep
    tr_index = np.concatenate(tr_index)
    te_index = np.concatenate(te_index)
    return tr_index,te_index


def train_val_test_split(mat_dir, kfold = 5, fold = 0, rep=100):

    subjects = loadmat(mat_dir)
    pat_id = list(subjects['subjects_pat'])
    con_id = list(subjects['subjects_con'])
    sub_list = pat_id + con_id
    label = list(np.ones(len(pat_id)))+list(np.zeros(len(con_id)))
    list1, list2 = (list(t) for t in zip(*sorted(zip(sub_list, label))))


    x_ind = range(len(sub_list))
    y_ind = np.array(list2)

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=666)
    skf2 = StratifiedKFold(n_splits=kfold-1, shuffle=True, random_state = 666)
    test_index = list()
    train_index = list()
    validation_index = list()

    i = 0
    for a, b in skf.split(x_ind, y_ind):
        test_index.append(b)
        temp1, temp2 = list(skf2.split(a, y_ind[a]))[0]
        c = a[temp1]
        d = a[temp2]
        train_index.append(c)
        validation_index.append(d)
        i = i + 1

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = validation_index[fold]

    tr_index = list()
    te_index = list()
    val_index = list()
    count = 0
    for i in range(len(sub_list)):
        if list2[i] ==1:
            rep1 = rep
        else:
            rep1 = rep
        ll = list(range(count,count+rep1))
        if i in train_id:
            tr_index.append(ll)
        elif i in test_id:
            te_index.append(ll)
        elif i in val_id:
            val_index.append(ll)
        count = count + rep1
    tr_index = np.concatenate(tr_index)
    te_index = np.concatenate(te_index)
    val_index = np.concatenate(val_index)
    return tr_index,te_index,val_index



