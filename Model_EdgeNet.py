from __future__ import print_function
from matplotlib import transforms
from sqlalchemy import false
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import cuda

def knn(x, k):

    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)#x**2的意思是x中的每一个数都平方
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx=idx.cuda()
    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :] #(2400,5)
    feature = feature.view(batch_size, num_points, k, num_dims)  #(20 , 24 , 5 , 5)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1) #(20 , 24 , 5 , 5)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() #(20 , 10 , 24 , 5)
  
    return feature

class STNkd(nn.Module): #pointnet的仿射变换(变换k维(k=5))
    def __init__(self, k=5):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]#batchsize=20 x=[20,5 num_dims,12 num_points] device=cuda
        x = F.relu(self.bn1(self.conv1(x)))#x=[20,64 num_channels,12 num_points]
        x = F.relu(self.bn2(self.conv2(x)))#x=[20,64 num_channels,12 num_points]
        x = F.relu(self.bn3(self.conv3(x)))#x=[20,128 num_channels,12 num_points]
        x = F.relu(self.bn4(self.conv4(x)))#x=[20,1024 num_channels,12 num_points]
        x = torch.max(x, 2, keepdim=True)[0]#x=[20,1024 num_channels,1]
        x = x.view(-1, 1024)#x=[20,1024]

        x = F.relu(self.bn5(self.fc1(x)))#x=[20,512 num_channels]
        x = F.relu(self.bn6(self.fc2(x)))#x=[20,256 num_channels]
        x = self.fc3(x)#x=[20,25]

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)#iden=[20,25]
        if x.is_cuda:
            iden = iden.cuda()#如果x是在GPU上的话，那么将iden转为GPU上的数据格式
        x = x + iden#x=[20,25]
        x = x.view(-1, self.k, self.k)#x=[20,5,5]
        return x

class PointNetfeat(nn.Module): #pointnet提取一帧中的全局特征，最后的最大池化变为Attention
    def __init__(self, global_feat = False, feature_transform = True):
        super(PointNetfeat, self).__init__()
        self.k=5
        self.stn = STNkd(5)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(1024)
        self.bn5 = nn.BatchNorm1d(48)
        self.bn6 = nn.BatchNorm1d(24)

        self.conv1 = nn.Sequential(nn.Conv2d(5*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2)) 
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 1024, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2)) 

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]#n_pts=12
        trans = self.stn(x)#trans[20 , 5 num_dims , 5 num_dims]  x[20 , 5 num_dims , 12 num_points] ALL cuda

        x = get_graph_feature(x, k=self.k)#x=[20 , 10 , 12 , 5]
        x=self.conv1(x)   #x=[20 , 64 , 12 , 5]
        x1 = x.max(dim=-1, keepdim=False)[0] # x1=[20 , 64 , 12]  #x=[20 , 64 , 12 , 5]         

        if self.feature_transform:
            trans_feat = self.fstn(x1)  # transfeat=[20 , 64 , 64]
            x1 = x1.transpose(2,1)  # x1=[20 , 12 , 64]
            x1 = torch.bmm(x1, trans_feat)  # x1=[20 , 12 , 64]
        else:
            trans_feat = None

        x1 =x1.permute(0,2,1)
        pointfeat = x1  # pointfeat=[20 , 64 , 12]

        x = get_graph_feature(x1, k=self.k) #x=[20,128,12,5]
        x=self.conv2(x) #x=[20,64,12,5]
        x2 = x.max(dim=-1, keepdim=False)[0] #x2=[20,64,12]        
        x = get_graph_feature(x2, k=self.k) #x=[20,128,12,5]
        x=self.conv3(x) #x=[20,128,12,5]
        x33 = x.max(dim=-1, keepdim=False)[0] #x=[20,128,12]
        x = get_graph_feature(x33, k=self.k) #x=[20,256,12,5]
        x=self.conv4(x) #x=[20,1024,12,5]
        x44 = x.max(dim=-1, keepdim=False)[0] #x=[20,1024,12]

        x4 = torch.max(x44, 2, keepdim=True)[0]#keepdim=True表示输出和输入的维度一样 #x=[20,1024,1]
        x4 = x4.view(-1, 1024) #x=[20,1024]

        if self.global_feat:
            return x4, trans, trans_feat
        else:
            x = x4.view(-1, 1024, 1).repeat(1, 1, n_pts)#所得x=[20,1024,12]
            x = torch.cat([pointfeat,  x2 , x33 , x], 1)#在拼接之后x[20 1280 12]
            return x, trans, trans_feat

class GlobalTiRNN(nn.Module):
    def __init__(self, points=12, k = 6, feature_transform=False):
        super(GlobalTiRNN, self).__init__()
        self.points=points
        self.k=k
        self.feature_transform=feature_transform
        
        self.conv1 = torch.nn.Conv1d(1280, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)



    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return  x

class GlobalTiModule(nn.Module):
    def __init__(self, points=12, k = 6, global_feat = False, feature_transform = True):
        super(GlobalTiModule, self).__init__()
        self.points=points
        self.pointnet=PointNetfeat(global_feat, feature_transform)#提取一帧中的特征
        self.grnn=GlobalTiRNN(points, k, feature_transform)

    def forward(self, x):#输入的原始x[20,5,12]
        x, _, _ =self.pointnet(x)#所得x[20,24,12]
        g_vec=self.grnn(x)
        return g_vec


def Cor_Pred(pred: cuda.FloatTensor,
             target: cuda.FloatTensor,
             nepoch
            ):
    """
    Input:pred[240 batchsize*npoints , 1 prediction]
    Output:[20 batchsize ,6 num_classes]
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    nepoch = nepoch
    x = np.zeros((nepoch*211*20,6))
    y = np.zeros((nepoch*211*20,6))
    for i in range(nepoch):
        for j in range(211):
            for k in range(20):
                for h in range(12):
                    if target[(211*20*12*i)+(20*12*j)+(12*k)+h]==pred[(211*20*12*i)+(20*12*j)+(12*k)+h]:
                      n=target[(211*20*12*i)+(20*12*j)+(12*k)+h].astype(int)
                      x[(211*20*i)+(20*j)+k,n] = x[(211*20*i)+(20*j)+k,n] + 1
                    if target[(211*20*12*i)+(20*12*j)+(12*k)+h]!=pred[(211*20*12*i)+(20*12*j)+(12*k)+h]:
                      n=target[(211*20*12*i)+(20*12*j)+(12*k)+h].astype(int)
                      y[(211*20*i)+(20*j)+k,n] = y[(211*20*i)+(20*j)+k,n] + 1
    return x, y

def Cor_Pred2(pred: cuda.FloatTensor,
             target: cuda.FloatTensor,
            ):
    """
    Input:pred[240 batchsize*npoints , 1 prediction]
    Output:[20 batchsize ,6 num_classes]
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()-1
    x = np.zeros((88*20,6))
    y = np.zeros((88*20,6))
    for j in range(88):
        for k in range(20):
            for h in range(12):
                if target[(20*12*j)+(12*k)+h]==pred[(20*12*j)+(12*k)+h]:
                    n=target[(20*12*j)+(12*k)+h].astype(int)
                    x[(20*j)+k,n] = x[(20*j)+k,n] + 1
                if target[(20*12*j)+(12*k)+h]!=pred[(20*12*j)+(12*k)+h]:
                    n=target[(20*12*j)+(12*k)+h].astype(int)
                    y[(20*j)+k,n] = y[(20*j)+k,n] + 1
    return x, y
