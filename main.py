import os
import numpy as np
import argparse
import time
import copy

import matplotlib.pyplot as plt
import deepdish as dd

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from BiopointData import BiopointDataset
from torch_geometric.data import DataLoader
from net.brain_networks import LI_Net,NNGAT_Net

from utils.utils import normal_transform_train,normal_transform_test,train_val_test_split
from utils.mmd_loss import MMD_loss


torch.manual_seed(123)

EPS = 1e-15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=100, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/praug_30/', help='root directory of the dataset')
parser.add_argument('--matroot', type=str, default='MAT/clear_subjects.mat', help='root directory of the subject ID')
parser.add_argument('--fold', type=int, default=1, help='training which fold')
parser.add_argument('--lr', type = float, default=0.01, help='learning rate')
parser.add_argument('--rep', type=int, default=30, help='augmentation times')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-2, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=1, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=1, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='s1 distance regularization')
parser.add_argument('--lamb4', type=float, default=0.1, help='s2 distance regularization')
parser.add_argument('--lamb5', type=float, default=0, help='s1 consistence regularization')
parser.add_argument('--lamb6', type=float, default=0, help='s2 consistence regularization')
parser.add_argument('--distL', type=str, default='bce', help='bce || mmd')
parser.add_argument('--poolmethod', type=str, default='topk', help='topk || sag')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam || SGD')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--nodes', type=int, default=84, help='number of nodes')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--net', type=str, default='NNGAT', help='model name: NNGAT || LI_NET')
parser.add_argument('--indim', type=int, default=84, help='feature dim')
parser.add_argument('--nclass', type=int, default=2, help='feature dim')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--normalization', action='store_true')
parser.set_defaults(save_model=True)
parser.set_defaults(normalization=True)
opt = parser.parse_args()




#################### Parameter Initialization #######################
name = 'Biopoint'
writer = SummaryWriter(os.path.join('./log/{}_fold{}_consis{}'.format(opt.net,opt.fold,opt.lamb5)))



############# Define Dataloader -- need costumize#####################
dataset = BiopointDataset(opt.dataroot, name)

############### split train, val, and test set -- need costumize########################
tr_index,te_index,val_index = train_val_test_split(mat_dir=opt.matroot,fold=opt.fold,rep = opt.rep)
########### skip these two lines for cv.. ############################
val_index = np.concatenate([te_index,val_index])
te_index = val_index
######################################################################
train_mask = torch.zeros(len(dataset), dtype=torch.uint8)
test_mask = torch.zeros(len(dataset), dtype=torch.uint8)
val_mask = torch.zeros(len(dataset), dtype=torch.uint8)
train_mask[tr_index] = 1
test_mask[te_index] = 1
val_mask[val_index] = 1
test_dataset = dataset[test_mask]
train_dataset = dataset[train_mask]
val_dataset = dataset[val_mask]


# ######################## Data Preprocessing ########################
# ###################### Normalize features ##########################
if opt.normalization:
    for i in range(train_dataset.data.x.shape[1]):
        train_dataset.data.x[:, i], lamb, xmean, xstd = normal_transform_train(train_dataset.data.x[:, i])
        test_dataset.data.x[:, i] = normal_transform_test(test_dataset.data.x[:, i],lamb, xmean, xstd)
        val_dataset.data.x[:, i] = normal_transform_test(val_dataset.data.x[:, i], lamb, xmean, xstd)

test_loader = DataLoader(test_dataset,batch_size=opt.batchSize,shuffle = False)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= True)



############### Define Graph Deep Learning Network ##########################
if opt.net =='LI_NET':
    model = LI_Net(opt.ratio).to(device)
elif opt.net == 'NNGAT':
    model = NNGAT_Net(opt.ratio, indim=opt.indim, poolmethod = opt.poolmethod).to(device)


print(model)
print('ground_truth: ', test_dataset.data.y, 'total: ', len(test_dataset.data.y), 'positive: ',sum(test_dataset.data.y))
if opt.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)


############################### Define Other Loss Functions ########################################

if opt.distL == 'bce':
    ###### bce loss
    def dist_loss(s,ratio):
        if ratio > 0.5:
            ratio = 1-ratio
        s = s.sort(dim=1).values
        res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
        return res

elif opt.distL == 'mmd':
    ######## mmd
    mmd = MMD_loss()
    def dist_loss(s,ratio):
        s = s.sort(dim=1).values
        source = s[:,-int(s.size(1)*ratio):]
        target = s[:,:int(s.size(1)*ratio)]
        res = mmd(source,target)
        return -res

def consist_loss(s):
    if len(s) == 0:
        return 0
    else:
        s = torch.sigmoid(s)
        W = torch.ones(s.shape[0],s.shape[0])
        D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
        L = D-W
        L = L.to(device)
        res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
        return res

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')
    model.train()

    s1_list = []
    s2_list = []
    loss_all = 0
    loss_en1_all  = 0
    loss_en2_all = 0

    i = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr)

        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())

        loss_c = F.nll_loss(output, data.y) # classification loss

        loss_dist1 = dist_loss(s1, opt.ratio)
        loss_dist2 = dist_loss(s2, opt.ratio)
        loss_consist = consist_loss(s1[data.y == 1]) + consist_loss(s1[data.y == 0])
        loss = opt.lamb0 * loss_c \
               + opt.lamb3 * loss_dist1 + opt.lamb4 * loss_dist2 + opt.lamb5 * loss_consist
        writer.add_scalar('train/classification_loss', loss_c, epoch * len(train_loader) + i)
        writer.add_scalar('train/entropy_loss1', loss_dist1, epoch * len(train_loader) + i)
        writer.add_scalar('train/entropy_loss2', loss_dist2, epoch * len(train_loader) + i)
        writer.add_scalar('train/consistance_loss', loss_consist, epoch * len(train_loader) + i)

        i = i + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        loss_en1_all +=loss_dist1.item() *data.num_graphs
        loss_en2_all += loss_dist2.item() * data.num_graphs
        optimizer.step()
        scheduler.step()

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)

        if not os.path.exists('outputs/'):
            os.makedirs('outputs/')
        if epoch%5 == 0:
            dd.io.save(
                'outputs/train_s1_{}_epoch{}_dist{}_cnsis{}_pool{}.h5'.format(opt.net, epoch, opt.lamb3, opt.lamb5,
                                                                              opt.ratio), {'s1': s1_arr})
            dd.io.save(
                'outputs/train_s2_{}_epoch{}_dist{}_cnsis{}_pool{}.h5'.format(opt.net, epoch, opt.lamb3, opt.lamb5,
                                                                              opt.ratio), {'s2': s1_arr})

    return loss_all / len(train_dataset), s1_arr, s2_arr, loss_en1_all / len(train_dataset),loss_en2_all / len(train_dataset)


###################### Network Testing Function#####################################
def test_acc(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output,_,_= model(data.x, data.edge_index, data.batch, data.edge_attr)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def test_loss(loader,epoch):
    print('testing...........')
    model.eval()
    loss_all = 0

    i=0
    for data in loader:
        data = data.to(device)
        output,s1,s2 = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss_c = F.nll_loss(output, data.y)

        loss_dist1 = dist_loss(s1, opt.ratio)
        loss_dist2 = dist_loss(s2, opt.ratio)
        loss_consist = consist_loss(s1)
        loss = opt.lamb0 * loss_c \
               + opt.lamb3 * loss_dist1 + opt.lamb4 * loss_dist2 + opt.lamb5 * loss_consist
        writer.add_scalar('val/classification_loss', loss_c, epoch * len(loader) + i)
        writer.add_scalar('val/entropy_loss1', loss_dist1, epoch * len(loader) + i)
        writer.add_scalar('val/entropy_loss2', loss_dist2, epoch * len(loader) + i)
        writer.add_scalar('val/consistance_loss', loss_consist, epoch * len(loader) + i)
        i = i + 1

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)

#######################################################################################
############################   Model Training #########################################
#######################################################################################
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10
for epoch in range(0, opt.n_epochs):
    since  = time.time()
    tr_loss, s1_arr, s2_arr,le1,le2 = train(epoch)
    tr_acc = test_acc(train_loader)
    val_acc = test_acc(val_loader)
    val_loss = test_loss(val_loader,epoch)
    time_elapsed = time.time() - since
    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, Test Loss: {:.7f}, Test Acc: {:.7f}'.format(epoch, tr_loss,
                                                       tr_acc, val_loss, val_acc))

    writer.add_scalars('Acc',{'train_acc':tr_acc,'val_acc':val_acc},  epoch)
    writer.add_scalars('Loss', {'train_loss': tr_loss, 'val_loss': val_loss},  epoch)
    writer.add_scalar('Ent/ent1', le1, epoch)
    writer.add_scalar('Ent/ent2', le2, epoch)
    writer.add_histogram('Hist/hist_s1', s1_arr, epoch)
    writer.add_histogram('Hist/hist_s2', s2_arr, epoch)


    if val_loss < best_loss and epoch > 5:
        print("saving best model")
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if opt.save_model:
            torch.save(best_model_wts,
                       'models/rep{}_biopoint_{}_{}_{}.pth'.format(opt.rep,opt.fold,opt.net,opt.lamb5))

#######################################################################################
######################### Testing on testing set ######################################
#######################################################################################
model.load_state_dict(best_model_wts)
model.eval()
test_accuracy = test_acc(test_loader)
test_l= test_loss(test_loader,0)
print("===========================")
print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy, test_l))
print(opt)



