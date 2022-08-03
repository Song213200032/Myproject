import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset
import random

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=12,
                 split='train'):
        self.root = root
        self.npoints=npoints
        self.fileList=[]
        renfiledirList=os.listdir(root)
        if(split=='train'):
            for renfiledir in range(int(len(renfiledirList)*0.7)):
                for filedir in os.listdir(os.path.join(root,renfiledirList[renfiledir])):
                    for file in os.listdir(os.path.join(root,renfiledirList[renfiledir],filedir)):
                        self.fileList.append(os.path.join(root,renfiledirList[renfiledir],filedir,file))
        if(split=="test"):
             for renfiledir in range(int(len(renfiledirList)*0.3)):
                for filedir in os.listdir(os.path.join(root,renfiledirList[-renfiledir])):
                    for file in os.listdir(os.path.join(root,renfiledirList[-renfiledir],filedir)):
                        self.fileList.append(os.path.join(root,renfiledirList[-renfiledir],filedir,file))

    def __getitem__(self, index):
        mat=loadmat(self.fileList[index])
        point_set = mat['xyzivb_ti'][:,0:5].astype(np.float32)
        seg = mat['xyzivb_ti'][:,5].astype(np.int64)
        seg_3 = mat['xyzivb_ti'][:,6].astype(np.int64)
        #从0-seg长度中选取npoints个数 组成一维数组
        seglen=len(seg)
        if(seglen<self.npoints):
            a=np.arange(seglen)
            bnumber=int((self.npoints-seglen)/seglen)
            b=np.random.choice(seglen,bnumber*seglen,replace=True)
            c=np.random.choice(seglen,(self.npoints-seglen-bnumber*seglen),replace=False)
            choice=np.hstack((a,b))
            choice=np.hstack((choice,c))
        else:
            choice=np.arange(seglen)
            choice =np.random.choice(choice,self.npoints,replace=False)
        #resample
        point_set = point_set[choice, :]
        
        #将point_set进行归一化，每一个维度减去该维度平均值，再除以绝对值最大的值
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        npoints=np.shape(point_set)[0]
        # if self.data_augmentation:
        #     theta = np.random.uniform(0,np.pi*2)
        #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        #     point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
        #     point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter
        seg = seg[choice]
        seg_3 =seg_3[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        seg_3 = torch.from_numpy(seg_3)

        
        # for chan_2_i in range(point_set.shape[0]):
        #     for chan_2_j in range(point_set.shape[0]-chan_2_i-1):
        #         if(point_set[chan_2_j][2]<point_set[chan_2_j+1][2]):
        #             exchange1=point_set[chan_2_j+1]
        #             point_set[chan_2_j+1]=point_set[chan_2_j]
        #             point_set[chan_2_j]=exchange1
        #             exchange2=seg[chan_2_j+1]
        #             seg[chan_2_j+1]=seg[chan_2_j]
        #             seg[chan_2_j]=exchange2


        # print(point_set.size())
        # print(seg.size())
        return point_set, seg ,seg_3

    def __len__(self):
        return len(self.fileList)

if __name__=='__main__':
    path='labeledData'
    dataset=ShapeNetDataset(path)
    points,seg=dataset[0]
    print(points.size())
    print(seg.size())
