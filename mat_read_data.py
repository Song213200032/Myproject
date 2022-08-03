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
from scipy.io import loadmat
import os
import os.path
import pandas as pd



def REVSIZE(EX1):
    '''
    Input:Tensor[-1,5]
    Output:Tensor[-1,12,5]
    '''
    length = EX1.shape[0]
    WIDTH = EX1.shape[1]
    EX1 = EX1.view(length,WIDTH)
    if length%60==0:
        EX1=EX1.view(-1,12,WIDTH)
    else:
        n=(length//60)+1
        m=n*60-length
        EX1=EX1.numpy()
        EXX=EX1
        EXXX=EX1[0:m,:]
        EX1=np.concatenate((EXX,EXXX),axis=0)
        EX1=torch.tensor(EX1)
        EX1=EX1.view(-1,12,WIDTH)
    return EX1


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


class MYATTENTION(nn.Module): #pointnet提取一帧中的全局特征，最后的最大池化变为Attention
    def __init__(self, global_feat = False, feature_transform = True):
        super(MYATTENTION, self).__init__()
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

if __name__=='__main__':
    stn = STNkd(3)
    mylstm = nn.LSTM(input_size=3, hidden_size=3, num_layers=64, batch_first=True, dropout=0.1, bidirectional=False)
    root='testmatlabdata_original'
    fileList=[]
    renfiledirList=os.listdir(root)
    for renfiledir in range(int(len(renfiledirList))):
        fileList.append(os.path.join(root,renfiledirList[renfiledir]))
    pppp1 = torch.zeros((0,3))
    for i in range(len(fileList)):
        mat=loadmat(fileList[i])
        points=np.zeros((0,3))
        seg=np.zeros((0,1))
        for j in range(10):
            points = np.concatenate((points,mat['b_xyz_kinect_C'][0,j]),axis=0).astype(np.float32)

    points = REVSIZE(torch.tensor(points)).transpose(2,1)
    batch = points.shape[0]
    h0 = torch.zeros((64, batch, 3), dtype=torch.float32, device='cpu')
    c0 = torch.zeros((64, batch, 3), dtype=torch.float32, device='cpu')
    ppp1 = stn(points)
    vec, (hn, cn) = mylstm(points.transpose(2,1), (h0, c0))
    points = points.transpose(2,1)
    ppp2 = torch.bmm(points,ppp1)
    ppp2 = ppp2.view(-1,3)
    vec = vec.contiguous().view(-1,3)
    ppp2 = ppp2.detach().numpy()
    vec = vec.detach().numpy()
    dp = pd.DataFrame(ppp2)
    dp2 = pd.DataFrame(vec)
    dp.to_csv('./testmatlabdata_processed/STN_result.csv')
    dp2.to_csv('./testmatlabdata_processed/LSTM_3_result.csv')
    

# file = '文件位置'
# #mat——dtype=True，保证了导入后变量的数据类型与原类型一致
# data = loadmat（file,mat_dtype=True）
# #导入后的data是一个字典，取出想要的变量字段即可
 