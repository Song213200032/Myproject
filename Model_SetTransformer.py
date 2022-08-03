import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import cuda
import numpy as np

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=5,#根据输入张量的第三维有几个元素来确定 ( X, , )
        num_outputs=12,#决定输出张量的第二维有几个元素 ( ,Y, )
        dim_output=6,#决定输出张量的第三维有几个元素 ( , , Z)
        num_inds=32,
        dim_hidden=64,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return F.softmax(self.dec(self.enc(X)),dim=2).squeeze()

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


if __name__=='__main__':
    a=torch.rand(1,2,3)
    print(a.shape[0])
    # testa=torch.rand(2,12,5)
    # TT=SetTransformer()
    # testb=TT(testa)
    # print(testa)
    # print(testb)
