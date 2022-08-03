from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from Model_Mixnet import *
from loaddata_Mixnet import ShapeNetDataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat 
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
writer =SummaryWriter('./Pointnet_3')


manualSeed = random.randint(1, 10000)  # fix seed

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batchSize=4 #一批处理四个帧
sequenceSize=5 #一帧
workers=0
outf='seg'
feature_transform=True
Model_Mixnet=''
nepoch=20 #一共训练几轮
nepoch1=10
nepoch2=10
nepoch3=10

npoints=12

path='labeledData'

dataset = ShapeNetDataset(root=path, npoints=npoints, split='train')
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchSize*sequenceSize,
    shuffle=True,
    num_workers=int(workers))

#shuffle:：是否将数据打乱
#num_workers：使用多进程加载的进程数，0代表不使用多进程

test_dataset = ShapeNetDataset(root=path, npoints=npoints, split='test')
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batchSize*sequenceSize,
    shuffle=True,
    num_workers=int(workers))

print(len(dataset), len(test_dataset))
#len()函数作用于矩阵时，返回的值是矩阵的行数
#len()函数作用于张量时，返回的是张量的第一维的大小
num_classes = 2
num_classes1 = 2
num_classes2 = 2
num_classes3 = 2
print('classes', num_classes)
try:
    os.makedirs(outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = GlobalTiModule(npoints, num_classes, feature_transform=feature_transform)
classifier1 = GlobalTiModule(npoints, 2, feature_transform=feature_transform)
classifier2 = GlobalTiModule(npoints, 2, feature_transform=feature_transform)
classifier3 = GlobalTiModule(npoints, 2, feature_transform=feature_transform)

if Model_Mixnet != '':
    classifier.load_state_dict(torch.load(Model_Mixnet))

optimizer = optim.Adam(classifier.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

optimizer1 = optim.Adam(classifier1.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

optimizer2 = optim.Adam(classifier2.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler2 = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

optimizer3 = optim.Adam(classifier3.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler3 = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

classifier.cuda()
classifier1.cuda()
classifier2.cuda()
classifier3.cuda()

num_batch = len(dataset) / (batchSize*sequenceSize)
#num_batch代表的是，在一轮训练中，共有多少批次数据需要处理
#batchSize*sequenceSize代表的是，一批数据中有多少个帧点云

if __name__=='__main__':
    tensor1 = torch.zeros(0).cuda()
    tensor2 = torch.zeros(0).cuda()
    tensor21 = torch.zeros(0).cuda()
    tensor22 = torch.zeros(0).cuda()
    POINTS = torch.zeros((0,5)).cuda()
    TARGET = torch.zeros((0,1)).cuda()
    POINTSS = torch.zeros((0,5)).cuda()
    TARGETT = torch.zeros((0,1)).cuda()
    for epoch in range(nepoch):
        scheduler.step() #更新optimizer中的学习率
        accADD=0
        for i, data in enumerate(dataloader, 0): #enumerate函数先返回下标i，后返回元素值data
            points,target,target2=data #data中包含两个变量，分别是point_set 以及seg(标签数据)#points=[20 batch ,24 num_points ,5 num_dims]; target=[20 batch , 24 num_points];
            if(points.size()[0]<batchSize*sequenceSize):#batchSize:4    seqenceSize:5
                continue
            points, target ,target2 = points.cuda(), target.cuda() , target2.cuda()#将CPU上的Tensor或变量放到GPU上
            points = points.transpose(2, 1)#points=[20 batch ,5 num_dims ,12 num_points];
            optimizer.zero_grad()
            classifier.train()
            pred= classifier(points)
            #pred - g_loc;   att - attn_weights;
            # h - hn;         c  -  cn;
            #可以把hn理解为当前时刻，LSTM层的输出结果，而cn是记忆单元中的值
            #g_vec则是包括当前时刻以及之前时刻所有hn的输出值
            #g_loc是经过全连接层的g_vec，分了下类，提取了一下特征
            #print(pred.size())   #pred[20,12,6]
            #print(target.size())
            pred = pred.view(-1, num_classes) #重构张量的维度（x，6）
            #经过打印发现.view()只是将不同的矩阵从三维按顺序拼接成了二维的形状
            target = target.view(-1, 1)[:, 0] - 1
            target2 = target2.view(-1, 1)[:, 0] - 1
            target2 = target2.resize(len(target2),1)
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            #print(pred.data)
            #print(pred.data.shape)
            pred_choice = pred.data.max(1)[1]

            POINTS = torch.cat((POINTS,points.transpose(2, 1).view(-1, 5)),0)
            TARGET = torch.cat((TARGET,target2),0)

            #torch.max(1)[1]， 只返回矩阵每一行最大值的每个索引
            #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            #print(pred_choice)
            tensor1 = torch.cat((tensor1,pred_choice),0)
            tensor2 = torch.cat((tensor2,target),0)

            correct = pred_choice.eq(target.data).cpu().sum()
            #pred_choice.eq(target.data)是比较pred_choice与target.data相同的元素
            #.cpu().sum()是把得到的相同的元素个数求和的意思
            #将GPU上的Tensor或变量放到CPU上
            #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            #print(correct)
            acc=(correct.item()/float(batchSize *sequenceSize* npoints))
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), acc))
            accADD+=acc
            if i % 10 == 0:#每训练十组，就测试一组
                j, data = next(enumerate(testdataloader, 0))#返回testdataloader的下一个项目
                points,target,target2=data
                points, target ,target2 = points.cuda(), target.cuda() ,target2.cuda()
                points = points.transpose(2, 1)
                for name, module in classifier.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.training = False
                pred= classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(batchSize *sequenceSize* npoints)))
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(accADD/211)

        tensor21 = torch.cat((tensor21,tensor1),0)
        tensor22 = torch.cat((tensor22,tensor2),0)
        tensor1 = torch.zeros(0).cuda()
        tensor2 = torch.zeros(0).cuda()
        POINTSS = torch.cat((POINTSS,POINTS),0)
        TARGETT = torch.cat((TARGETT,TARGET),0)
        TARGET = torch.zeros((0,1)).cuda()
        POINTS = torch.zeros((0,5)).cuda()

        writer.add_scalar('Acc', accADD/211, epoch)
        writer.add_scalar('TrainLoss', loss.item(), epoch)

        writer.close()

        torch.save(classifier.state_dict(), '%s/seg_Model_Mixnet_%d.pth' % (outf, epoch))

    x,y = Cor_Pred(tensor21,tensor22)
    ddx = pd.DataFrame(x)
    ddy = pd.DataFrame(y)
    ddx.to_csv('./Mixnet_cor_pred.csv')
    ddy.to_csv('./Mixnet_wor_pred.csv')  

    POINTS1,POINTS2,POINTS3,TARGET1,TARGET2,TARGET3,n1,n2,n3=REBATCH3(POINTSS,TARGETT,tensor21,tensor22)
    EX1=POINTS1
    EX11=TARGET1
    EX2=POINTS2
    EX22=TARGET2
    EX3=POINTS3
    EX33=TARGET3
    writer1 =SummaryWriter('./Mixnet_net1')
    for epoch in range(nepoch1):
        accAD1D=0
        for ii in range(n1-1):
            EX1=REVSIZE(EX1)
            EX11=REVSIZE(EX11)
            aa,bb,cc,dd,ee,ff=STAT(EX11)
            POINTS1=EX1.numpy()
            POINTS1=POINTS1[ii:ii+2,:,:]
            POINTS1=torch.tensor(POINTS1)
            POINTS1 = POINTS1.resize(2,POINTS1.shape[1],POINTS1.shape[2])
            POINTS1 = POINTS1.transpose(2,1).cuda().float()
            optimizer1.zero_grad()
            classifier1.train()
            pred1 = classifier1(POINTS1)
            pred1 = pred1.view(-1, num_classes1)
            EX11=EX11.view(-1,1)
            TARGET1 = EX11.numpy().astype(np.int64)
            TARGET1 = DATAPROCESS_3(TARGET1)
            TARGET1=TARGET1[ii*12:(ii+2)*12,:]
            TARGET1 = torch.tensor(TARGET1)
            TARGET1 = TARGET1.cuda()
            TARGET1 = TARGET1.squeeze()
            loss1 = F.nll_loss(pred1, TARGET1)
            loss1.backward()
            optimizer1.step()
            pred1_choice = pred1.data.max(1)[1]
            correct1 = pred1_choice.eq(TARGET1.data).cpu().sum()
            acc1=(correct1.item()/float(24))
            print('[%d: %d/%d] NET_1_train loss: %f accuracy: %f' % (epoch, ii, n1, loss1.item(), acc1))
            accAD1D+=acc1
        writer1.add_scalar('Acc', accAD1D/n1, epoch)
        writer1.add_scalar('TrainLoss', loss1.item(), epoch)

    writer2 =SummaryWriter('./Mixnet_net2')
    for epoch in range(nepoch2):
        accAD2D=0
        for iii in range(n2-1):
            EX2=REVSIZE(EX2)
            EX22=REVSIZE(EX22)
            POINTS2=EX2.numpy()
            POINTS2=POINTS2[iii:iii+2,:,:]
            POINTS2=torch.tensor(POINTS2)
            POINTS2 = POINTS2.resize(1,POINTS2.shape[0],POINTS2.shape[1])
            POINTS2 = POINTS2.transpose(2,1).cuda()
            optimizer2.zero_grad()
            classifier2.train()
            pred2 = classifier2(POINTS2)
            pred2 = pred2.view(-1, num_classes2)
            EX22=EX22.view(-1,1)
            TARGET2 = EX22.numpy() 
            TARGET2 = DATAPROCESS_3(TARGET2)
            TARGET2 = torch.tensor(TARGET2)
            loss2 = F.nll_loss(pred2, TARGET2)
            loss2.backward()
            optimizer2.step()
            pred2_choice = pred2.data.max(1)[1]
            correct2 = pred2_choice.eq(TARGET2.data).cpu().sum()
            acc2=(correct2.item()/float(2))
            print('[%d: %d/%d] NET_2_train loss: %f accuracy: %f' % (epoch, iii, n2, loss2.item(), acc2))
            accAD2D+=acc2
        writer2.add_scalar('Acc', accAD2D/n2, epoch)
        writer2.add_scalar('TrainLoss', loss2.item(), epoch)

    writer3 =SummaryWriter('./Mixnet_net3')
    for epoch in range(nepoch3):
        accAD3D=0
        for iiii in range(n3-1):
            EX3=REVSIZE(EX3)
            EX33=REVSIZE(EX33)
            POINTS3=EX3.numpy()
            POINTS3=POINTS3[iiii:iiii+2,:,:]
            POINTS3=torch.tensor(POINTS3)
            POINTS3 = POINTS3.resize(1,POINTS3.shape[0],POINTS3.shape[1])
            POINTS3 = POINTS3.transpose(2,1)
            optimizer3.zero_grad()
            classifier3.train()
            pred3 = classifier3(POINTS3)
            pred3 = pred3.view(-1, num_classes3)
            EX33=EX33.view(-1,1)
            TARGET3 = EX33.numpy() 
            TARGET3 = DATAPROCESS_3(TARGET3)
            TARGET3 = torch.tensor(TARGET3)
            loss3 = F.nll_loss(pred3, TARGET3)
            loss3.backward()
            optimizer3.step()
            pred3_choice = pred3.data.max(1)[1]
            correct3 = pred3_choice.eq(TARGET3.data).cpu().sum()
            acc3=(correct3.item()/float(1))
            print('[%d: %d/%d] NET_3_train loss: %f accuracy: %f' % (epoch, iiii, n3, loss3.item(), acc3))
            accADD+=acc3
        writer3.add_scalar('Acc', accAD3D/n3, epoch)
        writer3.add_scalar('TrainLoss', loss3.item(), epoch)

    print('MixNet_train loss: %f accuracy: %f' % (loss1.item()+loss2.item()+loss3.item(), accAD3D/n3+accAD2D/n2+accAD1D/n1))

    ##模型训练完之后，最后分割一遍，测试效果
    shape_ious = []
    POINTSS = torch.zeros((0,0)).cuda()
    TARGETT = torch.zeros((0,0)).cuda()
    for i,data in enumerate(tqdm(testdataloader, 0)):
        points,target=data
        if(points.size()[0]<batchSize*sequenceSize):
                continue
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()              
        classifier = classifier.eval()
        pred= classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1, 1)[:, 0] - 1
        POINTSS = torch.cat((POINTSS,points),0)
        TARGETT = torch.cat((TARGET,target),0)

    # pred_np = pred_choice.cpu().data.numpy()
    # target_np = target.cpu().data.numpy() - 1

    POINTS1,POINTS2,POINTS3,TARGET1,TARGET2,TARGET3,n1,n2,n3=REBATCH3(POINTSS,TARGETT,tensor21,tensor22)
    EX1=POINTS1
    EX11=TARGET1
    EX2=POINTS2
    EX22=TARGET2
    EX3=POINTS3
    EX33=TARGET3
    accAD1D=0
    for ii in range(n1):
        EX1=REVSIZE(EX1)
        POINTS1=EX1.numpy()
        POINTS1=POINTS1[ii,:]
        POINTS1=torch.tensor(POINTS1)
        POINTS1 = POINTS1.transpose(2,1)
        POINTS1 = POINTS1.resize(1,POINTS1.shape[0],POINTS1.shape[1])
        optimizer1.zero_grad()
        classifier1.train()
        pred1 = classifier1(POINTS1)
        pred1 = pred1.view(-1, num_classes)
        TARGET1 = EX11.numpy() - 1
        loss1 = F.nll_loss(pred1, TARGET1)
        loss1.backward()
        optimizer1.step()
        pred1_choice = pred1.data.max(1)[1]
        correct1 = pred1_choice.eq(TARGET1.data).cpu().sum()
        acc1=(correct1.item()/float(1))
        print('[%d: %d/%d] NET_1_train loss: %f accuracy: %f' % (epoch, ii, n1, loss1.item(), acc1))
        accAD1D+=acc1
        pred_choice = pred1.data.max(1)[1]
        pred_np = pred_choice.cpu().data.numpy()
        target_np = TARGET1.cpu().data.numpy() - 1
        for shape_idx in range(target_np.shape[0]):
            parts1 = range(2)#np.unique(target_np[shape_idx])
            part1_ious = []
            for part in parts1:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part1_ious.append(iou)
            shape_ious.append(np.mean(part1_ious))
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(accAD1D/n1)
    accAD2D=0
    for iii in range(n2):
        EX2=REVSIZE(EX2)
        POINTS2=EX2.numpy()
        POINTS2=POINTS2[ii,:]
        POINTS2=torch.tensor(POINTS2)
        POINTS2 = POINTS2.transpose(2,1)
        POINTS2 = POINTS2.resize(1,POINTS2.shape[0],POINTS2.shape[1])
        optimizer2.zero_grad()
        classifier2.train()
        pred2 = classifier2(POINTS2)
        pred2 = pred2.view(-1, num_classes)
        TARGET2 = EX22.numpy() - 1
        loss2 = F.nll_loss(pred2, TARGET2)
        loss2.backward()
        optimizer2.step()
        pred2_choice = pred2.data.max(1)[1]
        correct2 = pred2_choice.eq(TARGET2.data).cpu().sum()
        acc2=(correct2.item()/float(1))
        print('[%d: %d/%d] NET_2_train loss: %f accuracy: %f' % (epoch, iii, n2, loss2.item(), acc2))
        accAD2D+=acc2
        pred_choice = pred2.data.max(1)[1]
        pred_np = pred_choice.cpu().data.numpy()
        target_np = TARGET2.cpu().data.numpy() - 1
        for shape_idx in range(target_np.shape[0]):
            parts2 = range(2)#np.unique(target_np[shape_idx])
            part2_ious = []
            for part in parts2:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part2_ious.append(iou)
            shape_ious.append(np.mean(part2_ious))
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(accAD2D/n2)
    accAD3D=0
    for iiii in range(n3):
        EX3=REVSIZE(EX3)
        POINTS3=EX3.numpy()
        POINTS3=POINTS3[ii,:]
        POINTS3=torch.tensor(POINTS3)
        POINTS3 = POINTS3.transpose(2,1)
        POINTS3 = POINTS3.resize(1,POINTS3.shape[0],POINTS3.shape[1])
        optimizer3.zero_grad()
        classifier3.train()
        pred3 = classifier3(POINTS3)
        pred3 = pred3.view(-1, num_classes)
        TARGET3 = EX33.numpy() - 1
        loss3 = F.nll_loss(pred3, TARGET3)
        loss3.backward()
        optimizer3.step()
        pred3_choice = pred3.data.max(1)[1]
        correct3 = pred3_choice.eq(TARGET3.data).cpu().sum()
        acc3=(correct3.item()/float(1))
        print('[%d: %d/%d] NET_3_train loss: %f accuracy: %f' % (epoch, iiii, n3, loss3.item(), acc3))
        accADD+=acc3
        pred_choice = pred3.data.max(1)[1]
        pred_np = pred_choice.cpu().data.numpy()
        target_np = TARGET3.cpu().data.numpy() - 1
        for shape_idx in range(target_np.shape[0]):
            parts3 = range(2)#np.unique(target_np[shape_idx])
            part3_ious = []
            for part in parts3:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part3_ious.append(iou)
            shape_ious.append(np.mean(part3_ious))
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(accAD3D/n3)

    print('accuracy rate : {}'.format(accAD3D/n3+accAD2D/n2+accAD1D/n1))

    print("mIOU : {}".format( np.mean(shape_ious)))