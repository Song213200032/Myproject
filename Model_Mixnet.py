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
import pandas as pd

def REDATA3(pred:cuda.FloatTensor,col:int):
    '''
    Rebatch the data so as to classify secondly
    Input:[N_dims X] (with 3 kinds of data combined together)
    Output:[n1+n2+n3 X] (n1+n2+n3=N)
    '''
    pred = pred.cpu().numpy()
    width = pred.shape[1]
    leng = len(pred)
    n1=0
    n2=0
    n3=0
    for i in range(leng):
        if(pred[i,col]==0):
            n1+=1
        if(pred[i,col]==1):
            n2+=1
        if(pred[i,col]==2):
            n3+=1
    pred1=np.zeros((n1,width))
    pred2=np.zeros((n2,width))
    pred3=np.zeros((n3,width))
    j1=0
    j2=0
    j3=0
    for k1 in range(leng):
        n=pred[k1,col].astype(int)
        if(n==0):
            for kk1 in range(width):
                pred1[j1,kk1]=pred[k1,kk1]
            j1=j1+1
        if(n==1):
            for kk2 in range(width):
               pred2[j2,kk2]=pred[k1,kk2]
            j2=j2+1
        if(n==2):
            for kk3 in range(width):
               pred3[j3,kk3]=pred[k1,kk3]
            j3=j3+1
    x=np.concatenate((pred1,pred2,pred3),axis=0)
    x=torch.tensor(x).cuda()
    pred1=torch.tensor(pred1).cuda()
    pred2=torch.tensor(pred2).cuda()
    pred3=torch.tensor(pred3).cuda()

    return x,pred1,n1,pred2,n2,pred3,n3

def REBATCH3( POINTSS , TARGETT , tensor21 , tensor22 ):
    '''
    To re batch the data to 3 different parts
    Input:  np.array & tensor
    Output: tensor
    '''
    ##数据预处理
    tensor21 = tensor21.view(-1,1)
    tensor22 = tensor22.view(-1,1)
    POINTSS = torch.tensor(POINTSS)
    TARGETT = torch.tensor(TARGETT)
    POINTSS = torch.cat((POINTSS,tensor21),1)
    TARGETT = torch.cat((TARGETT,tensor21),1)
    ##排序
    POINTSS , POINTS1 , n1 , POINTS2 , n2 , POINTS3 , n3  = REDATA3(POINTSS,5)
    TARGETT , TARGET1 , BpP , TARGET2 , CpP , TARGET3 , DpP  = REDATA3(TARGETT,1)
    ##分割
    POINTS1 = POINTS1.cpu().numpy()
    POINTS1 = POINTS1[:,0:5]
    POINTS1 = torch.tensor(POINTS1)
    POINTS2 = POINTS2.cpu().numpy()
    POINTS2 = POINTS2[:,0:5]
    POINTS2 = torch.tensor(POINTS2)
    POINTS3 = POINTS3.cpu().numpy()
    POINTS3 = POINTS3[:,0:5]
    POINTS3 = torch.tensor(POINTS3)
    TARGET1 = TARGET1.cpu().numpy()
    TARGET1 = TARGET1[:,0]
    TARGET1 = torch.tensor(TARGET1)
    TARGET1 = TARGET1.view(-1,1)
    TARGET2 = TARGET2.cpu().numpy()
    TARGET2 = TARGET2[:,0]
    TARGET2 = torch.tensor(TARGET2)
    TARGET2 = TARGET2.view(-1,1)
    TARGET3 = TARGET3.cpu().numpy()
    TARGET3 = TARGET3[:,0]
    TARGET3 = torch.tensor(TARGET3)
    TARGET3 = TARGET3.view(-1,1)
    return POINTS1,POINTS2,POINTS3,TARGET1,TARGET2,TARGET3,n1,n2,n3

# def Cor_Pred(pred: cuda.FloatTensor,
#              target: cuda.FloatTensor
#             ):
#     """
#     Input:pred[240 batchsize*npoints , 1 prediction]
#     Output:[20 batchsize ,6 num_classes]
#     """
#     pred = pred.cpu().numpy()
#     target = target.cpu().numpy()
#     x = np.zeros((84400,3))
#     y = np.zeros((84400,3))
#     for i in range(20):
#         for j in range(211):
#             for k in range(20):
#                 for h in range(12):
#                     if target[(211*20*12*i)+(20*12*j)+(12*k)+h]==pred[(211*20*12*i)+(20*12*j)+(12*k)+h]:
#                       n=target[(211*20*12*i)+(20*12*j)+(12*k)+h].astype(int)
#                       x[(211*20*i)+(20*j)+k,n] = x[(211*20*i)+(20*j)+k,n] + 1
#                     if target[(211*20*12*i)+(20*12*j)+(12*k)+h]!=pred[(211*20*12*i)+(20*12*j)+(12*k)+h]:
#                       n=target[(211*20*12*i)+(20*12*j)+(12*k)+h].astype(int)
#                       y[(211*20*i)+(20*j)+k,n] = y[(211*20*i)+(20*j)+k,n] + 1
#     return x, y

def Cor_Pred(pred: cuda.FloatTensor,
             target: cuda.FloatTensor
            ):
    """
    Input:pred[240 batchsize*npoints , 1 prediction]
    Output:[20 batchsize ,6 num_classes]
    """
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    x = np.zeros((42200,3))
    y = np.zeros((42200,3))
    for i in range(2):
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

def DATAPROCESS_3(x):
    length = x.shape[0]
    for i in range(length):
        if(x[i,0]==0):
            x[i,0]=0
        if(x[i,0]==1):
            x[i,0]=0
        if(x[i,0]==2):
            x[i,0]=1
        if(x[i,0]==3):
            x[i,0]=2
        if(x[i,0]==4):
            x[i,0]=1
        if(x[i,0]==5):
            x[i,0]=2
    return x

def STAT(X):
    X = X.view(-1,1)
    x=(X).cpu().numpy().astype(int)
    length = len(x)
    a=0
    b=0
    c=0
    d=0
    e=0
    f=0
    for i in range(length):
        if(x[i]==0):
            a=a+1
        if(x[i]==1):
            b=b+1
        if(x[i]==2):
            c=c+1
        if(x[i]==3):
            d=d+1
        if(x[i]==4):
            e=e+1
        if(x[i]==5):
            f=f+1
    return a, b , c, d, e,f

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
        batchsize = x.size()[0]#batchsize=20 x=[20,5 num_dims,24 num_points] device=cuda
        x = F.relu(self.bn1(self.conv1(x)))#x=[20,12 num_channels,24 num_points]
        x = F.relu(self.bn2(self.conv2(x)))#x=[20,24 num_channels,24 num_points]
        x = F.relu(self.bn3(self.conv3(x)))#x=[20,192 num_channels,24 num_points]
        x = torch.max(x, 2, keepdim=True)[0]#x=[20,192 num_channels,1]
        x = x.view(-1, 192)#x=[20,192]

        x = F.relu(self.bn4(self.fc1(x)))#x=[20,96 num_channels]
        x = F.relu(self.bn5(self.fc2(x)))#x=[20,48 num_channels]
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
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.bn5 = nn.BatchNorm1d(48)
        self.bn6 = nn.BatchNorm1d(24)

        self.conv1 = torch.nn.Conv1d(5, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=12)

    def forward(self, x):
        n_pts = x.size()[2]#n_pts=24
        trans = self.stn(x)#trans[20 , 5 num_dims , 5 num_dims]  x[20 , 5 num_dims , 24 num_points] ALL cuda

        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))     

        if self.feature_transform:
            x = x.transpose(2,1)
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))#若keepdim=False则表示输出的维度被压缩了，也就是输出会比输入低一个维度
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]#keepdim=True表示输出和输入的维度一样
        x = x.view(-1, 1024)

        #print(x.size())
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)#所得x=[20,192,12]
            x = torch.cat([pointfeat, x], 1)#在拼接之后x[20 204 12]
            return x, trans, trans_feat

class GlobalTiRNN(nn.Module):
    def __init__(self, points=12, k = 3, feature_transform=False):
        super(GlobalTiRNN, self).__init__()
        self.points=points
        self.k=k
        self.feature_transform=feature_transform
        
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return  x

class GlobalTiModule(nn.Module):
    
    def __init__(self, points=12, k = 3, global_feat = False, feature_transform = True):
        super(GlobalTiModule, self).__init__()
        self.points=points
        self.pointnet=PointNetfeat(global_feat, feature_transform)#提取一帧中的特征
        self.grnn=GlobalTiRNN(points, k, feature_transform)

    def forward(self, x):#输入的原始x[20,5,12]
        x, _, _ =self.pointnet(x)#所得x[20,24,12]
        g_vec=self.grnn(x)
        return g_vec

if __name__=='__main__':
    x=np.random.randint(0,5,size=[10,1])
    x[9:11,0]
    print(x)
    x=DATAPROCESS_3(x)
    print(x)
    # pred = torch.randint(1,4,(2,8,6))
    # pred = torch.tensor(pred)
    # print(pred)
    # print(1/pred.shape[0])
    # pred=pred.data.max(2)[0]
    # print(pred)
    # predd = pred.numpy()
    # predd =predd -1
    # predd = predd[:,0:2]
    # print(predd)
    # a,bb,b,cc,c,dd,d=REDATA(pred,5)
    # print(a)
    # print(b)
    # print(c)
    # print(d)
    # s=np.zeros((2,2))
    # torch.tensor(s)
    # print(s)
