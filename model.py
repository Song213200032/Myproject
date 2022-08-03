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

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


#GlobalTiModule是最后的网络

class STN3d(nn.Module): #pointnet的仿射变换(只转xyz)
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module): #pointnet的仿射变换(变换k维(k=5))
    def __init__(self, k=5):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 12, 1)
        self.conv2 = torch.nn.Conv1d(12, 24, 1)
        self.conv3 = torch.nn.Conv1d(24, 192, 1)
        self.fc1 = nn.Linear(192, 96)
        self.fc2 = nn.Linear(96, 48)
        self.fc3 = nn.Linear(48, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(24)
        self.bn3 = nn.BatchNorm1d(192)
        self.bn4 = nn.BatchNorm1d(96)
        self.bn5 = nn.BatchNorm1d(48)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]#batchsize=20 x=[20,5,12]
        x = F.relu(self.bn1(self.conv1(x)))#x=[20,12,12]
        x = F.relu(self.bn2(self.conv2(x)))#x=[20,24,12]
        x = F.relu(self.bn3(self.conv3(x)))#x=[20,192,12]
        x = torch.max(x, 2, keepdim=True)[0]#x=[20,192,1]
        x = x.view(-1, 192)#x=[20,192]

        x = F.relu(self.bn4(self.fc1(x)))#x=[20,96]
        x = F.relu(self.bn5(self.fc2(x)))#x=[20,48]
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
        self.stn = STNkd(5)
        self.conv1 = torch.nn.Conv1d(5, 12, 1)
        self.conv2 = torch.nn.Conv1d(12, 24, 1)
        self.conv3 = torch.nn.Conv1d(24, 192, 1)
        self.conv4 = torch.nn.Conv1d(204, 96, 1)
        self.conv5 = torch.nn.Conv1d(96, 48, 1)
        self.conv6 = torch.nn.Conv1d(48, 24, 1)

        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(24)
        self.bn3 = nn.BatchNorm1d(192)
        self.bn4 = nn.BatchNorm1d(96)
        self.bn5 = nn.BatchNorm1d(48)
        self.bn6 = nn.BatchNorm1d(24)

        self.attn1=nn.Linear(192,48) #注意力机制
        self.attn2=nn.Linear(48,1)
        self.softmax=nn.Softmax(dim=1) #归一化为比重
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=12)

    def forward(self, x):
        n_pts = x.size()[2]#n_pts=12
        trans = self.stn(x)#trans[20 5 5] x[20 5 12]
        x = x.transpose(2, 1)#x[20 12 5]
        x = torch.bmm(x, trans)#经过矩阵相乘之后维度为x[20 12 5]
        x = x.transpose(2, 1)#x[20 5 12]
        x = F.relu(self.bn1(self.conv1(x)))#x[20 12 12]

        if self.feature_transform:
            trans_feat = self.fstn(x)#trans_feat=[20,12,12]
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x# pointfeat [20 12 12]
        x = F.relu(self.bn2(self.conv2(x)))# x=[20,24,12]
        x = self.bn3(self.conv3(x)) #x:[20,192,12]

        x = x.transpose(1,2) #x:[20,12,192]
        #print(x.size())
        attn_weights=self.softmax(self.attn2(self.attn1(x))) #attn_weights:[20,12,1]
        #print(attn_weights.size())
        attn_vec=torch.sum(x*attn_weights,dim=1) #attn_vec:[20,192]
        #print(attn_vec.size())
        attn_vec=attn_vec.unsqueeze(-1) #attn_vec:[20,192,1]
        #print(attn_vec.size())

        x = attn_vec.view(-1, 192) #[20 192]
        #print(x.size())
        if self.global_feat:
            return x, trans, trans_feat, attn_weights
        else:
            x = x.view(-1, 192, 1).repeat(1, 1, n_pts)#所得x=[20,192,12]
            x = torch.cat([pointfeat, x], 1)#在拼接之后x[20 204 12]
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))#最后x[20 24 12]
            return x, trans, trans_feat, attn_weights

class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

class GlobalTiRNN(nn.Module):
    def __init__(self, points=12, k = 6, feature_transform=False):
        super(GlobalTiRNN, self).__init__()
        self.points=points
        self.k=k
        self.feature_transform=feature_transform
        self.rnn=nn.LSTM(input_size=24, hidden_size=24, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)##修改前input为288，hidden为288
       
        self.bn1 = nn.BatchNorm1d(36)
        self.bn2=nn.BatchNorm1d(24)
        self.bn3=nn.BatchNorm1d(12)
        self.bn4=nn.BatchNorm1d(6)
       
        self.conv1 = torch.nn.Conv1d(24, 36, 1)
        self.conv3=torch.nn.Conv1d(36,24,1)
        self.conv4=torch.nn.Conv1d(24,12,1)
        self.conv2 = torch.nn.Conv1d(12, self.k, 1)



    def forward(self, x, h0, c0):
        batchsize =4########### x.size()[0]#batchsize=4                                                         
        n_frames =5 ########### x.size()[1]#n_frames=5
        vec, (hn, cn) = self.rnn(x, (h0, c0))##输入x=[20,12,24]
        #所得vec=[20,12,24]
        vec = vec.reshape(batchsize*n_frames, self.points, -1)
        #所得vec=[20,12,24]
        x = vec.transpose(2,1)
        #所得x=[20,24,12]
        x = F.relu(self.bn1(self.conv1(x)))#所得x=[20,12,12]
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn3(self.conv4(x)))
        x = F.relu(self.bn4(self.conv2(x)))#所得x=[20,6,12]
        x = x.transpose(2,1).contiguous()#所得x=[20,12,6]
        x = F.log_softmax(x.view(-1,self.k), dim=-1)#所得x=[240,6]
        x = x.view(batchsize*n_frames, self.points, self.k)#所得x=[20,12,6]
        return vec, x, hn, cn  

class GlobalTiModule(nn.Module):
    def __init__(self, points=12, k = 6, global_feat = False, feature_transform = True):
        super(GlobalTiModule, self).__init__()
        self.points=points
        self.pointnet=PointNetfeat(global_feat, feature_transform)#提取一帧中的特征
        self.grnn=GlobalTiRNN(points, k, feature_transform)

    def forward(self, x, h0, c0,  batch_size, length_size):#输入的原始x[20,5,12]
        x, _, _, attn_weights=self.pointnet(x)#所得x[20,24,12]
        x=x.transpose(2,1)#x=[20,12,24]
        x=x.transpose(0,1)#x=[12,20,24]
        #x=x.reshape(-1, self.points,24)#所得x[20,12,24]
        g_vec, g_loc, hn, cn=self.grnn(x, h0, c0)
        return g_vec, g_loc, attn_weights, hn, cn

# class PointNetfeatCBAM(nn.Module):
#     def __init__(self, global_feat = True, feature_transform = False):
#         super(PointNetfeatCBAM, self).__init__()
#         self.stn = STN3d()
#         self.attmaxconv1=torch.nn.Conv1d(3,3,1)
#         self.attmaxconv2=torch.nn.Conv1d(3,3,1)
#         self.attavgconv1=torch.nn.Conv1d(3,3,1)
#         self.attavgconv2=torch.nn.Conv1d(3,3,1)
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.global_feat = global_feat
#         self.feature_transform = feature_transform
#         if self.feature_transform:
#             self.fstn = STNkd(k=64)

#     def forward(self, x):
#         n_pts = x.size()[2]
#         trans = self.stn(x)
#         x = x.transpose(2, 1)
#         x = torch.bmm(x, trans)
#         print("x=={}".format(x[0:5,:,:]))

#         Amax=torch.max(x,1,keepdim=True)[0]
#         Aavg=x.mean(1,keepdim=True)
#         Amax=Amax.transpose(2,1)
#         Aavg=Aavg.transpose(2,1)
#         Amax = F.relu(self.attmaxconv1(Amax))
#         Amax = F.relu(self.attmaxconv2(Amax))
#         Aavg=F.relu(self.attavgconv1(Aavg))
#         Aavg=F.relu(self.attavgconv2(Aavg))
#         Att=torch.sigmoid(Amax+Aavg)
#         Att=Att.transpose(2,1)
#         x=x*Att

#         x = x.transpose(2, 1)
#         x = F.relu(self.bn1(self.conv1(x)))

#         if self.feature_transform:
#             trans_feat = self.fstn(x)
#             x = x.transpose(2,1)
#             x = torch.bmm(x, trans_feat)
#             x = x.transpose(2,1)
#         else:
#             trans_feat = None

#         pointfeat = x
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.bn3(self.conv3(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 1024)
#         if self.global_feat:
#             return x, trans, trans_feat
#         else:
#             x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
#             return torch.cat([x, pointfeat], 1), trans, trans_feat

if __name__ == '__main__':
    # sim_data = Variable(torch.rand(32,5,2500))
    # trans = STN3d()
    # out = trans(sim_data)
    # print('stn', out.size())
    # print('loss', feature_transform_regularizer(out))

    # sim_data_64d = Variable(torch.rand(32, 64, 2500))
    # trans = STNkd(k=64)
    # out = trans(sim_data_64d)
    # print('stn64d', out.size())
    # print('loss', feature_transform_regularizer(out))

    # pointfeat = PointNetfeat(global_feat=True)
    # out, _, _ = pointfeat(sim_data)
    # print('global feat', out.size())

    # pointfeat = PointNetfeat(global_feat=False)
    # out, _, _ = pointfeat(sim_data)
    # print('point feat', out.size())

    # cls = PointNetCls(k = 5)
    # out, _, _ = cls(sim_data)
    # print('class', out.size())

    #sim_data = Variable(torch.rand(32,5,12)) # (4*8,5,12) (batchsize*frames, dims, points)
    #seg = GlobalTiModule(k = 6)
    #h0=torch.zeros((3, 4, 1536), dtype=torch.float32, device='cpu')
    #c0=torch.zeros((3, 4, 1536), dtype=torch.float32, device='cpu')
    #vec, loc, attn_weights, hn, cn = seg(sim_data,h0,c0,4,8)
    #print('seg', vec.size())
    #print(loc.size())
    #print(attn_weights.size())
    #print(hn.size())
    #print(cn.size())
    #print(loc)

    h0=torch.zeros((3, 4, 1536), dtype=torch.float32, device='cpu')
    c0=torch.zeros((3, 4, 1536), dtype=torch.float32, device='cpu')
    data=Variable(torch.rand(4,8,12*24)) # (4*8,5,12) (batchsize*frames, dims, points)
    lstm=GlobalTiRNN()
    y1,(y2,y3)=lstm(data,h0,c0)
    #print(y1.size())
    #print(y2.size())
    #print(y3.size())
    #print(y4.size())



