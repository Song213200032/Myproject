from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from Model_mmMesh import *
from Model_mmMesh2 import *
from loaddata import ShapeNetDataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.io import loadmat 
from torch.utils.tensorboard import SummaryWriter
writer =SummaryWriter('./mmMesh_mixed62')


manualSeed = random.randint(1, 10000)  # fix seed

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batchSize=4 #一批处理四个帧
sequenceSize=5 #一帧
workers=0
outf='seg'
feature_transform=True
Model_mmMesh=''
nepoch=20 #一共训练几轮
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
num_classes = 6
print('classes', num_classes)
try:
    os.makedirs(outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = mmWaveModel()
classifier2 = mmWaveModel2()

if Model_mmMesh != '':
    classifier.load_state_dict(torch.load(Model_mmMesh))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
optimizer2 = optim.Adam(classifier2.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)

classifier.cuda()
classifier2.cuda()

h0=torch.zeros((3, batchSize, 64), dtype=torch.float32).cuda()
c0=torch.zeros((3, batchSize, 64), dtype=torch.float32).cuda()

num_batch = len(dataset) / (batchSize*sequenceSize)
#num_batch代表的是，在一轮训练中，共有多少批次数据需要处理
#batchSize*sequenceSize代表的是，一批数据中有多少个帧点云

if __name__=='__main__':
    tensor1 = torch.zeros(0).cuda()
    tensor2 = torch.zeros(0).cuda()
    tensor21 = torch.zeros(0).cuda()
    tensor22 = torch.zeros(0).cuda()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(nepoch):
        scheduler.step() #更新optimizer中的学习率
        scheduler2.step()
        accADD=0
        for i, data in enumerate(dataloader, 0): #enumerate函数先返回下标i，后返回元素值data
            points,target,target_2=data #data中包含两个变量，分别是point_set 以及seg(标签数据)#points=[20 batch ,24 num_points ,5 num_dims]; target=[20 batch , 24 num_points];
            if(points.size()[0]<batchSize*sequenceSize):#batchSize:4    seqenceSize:5
                continue
            points, target ,target_2= points.cuda(), target.cuda() ,target_2.cuda()#将CPU上的Tensor或变量放到GPU上
            #points = points.transpose(2, 1)#points=[20 batch ,5 num_dims ,12 num_points];
            points = torch.split(points,5,dim=0)
            oppo1 = points[0]
            oppo2 = points[1]
            oppo3 = points[2]
            oppo4 = points[3]
            oppo1 =oppo1.view(1,5,12,5)
            oppo2 =oppo2.view(1,5,12,5)
            oppo3 =oppo3.view(1,5,12,5)
            oppo4 =oppo4.view(1,5,12,5)
            points = torch.cat([oppo1,oppo2,oppo3,oppo4],dim=0)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            classifier.train()
            classifier2.train()
            pred= classifier(points,h0,c0,h0,c0)
            predd= classifier2(points,h0,c0,h0,c0)
            #pred - g_loc;   att - attn_weights;
            # h - hn;         c  -  cn;
            #可以把hn理解为当前时刻，LSTM层的输出结果，而cn是记忆单元中的值
            #g_vec则是包括当前时刻以及之前时刻所有hn的输出值
            #g_loc是经过全连接层的g_vec，分了下类，提取了一下特征
            #print(pred.size())   #pred[20,12,6]
            #print(target.size())
            pred = pred.view(-1, num_classes) #重构张量的维度（x，6）
            predd = predd.view(-1, 2) #重构张量的维度（x，6）
            #经过打印发现.view()只是将不同的矩阵从三维按顺序拼接成了二维的形状
            target_2 = target_2.view(-1, 1)[:, 0] - 1
            target = target.view(-1, 1)[:, 0] - 1
            # target = target.view(240,1)
            loss2 = F.nll_loss(predd, target_2)
            loss = F.nll_loss(pred, target)
            c_1 = loss.item()/(loss2.item()+loss.item())
            loss3=(1-c_1)*loss+c_1*loss2
            loss3.backward()
            optimizer.step()
            optimizer2.step()
            #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            #print(pred.data)
            #print(pred.data.shape)
            pred_choice = pred.data.max(1)[1]
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
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss3.item(), acc))
            accADD+=acc
            if i % 10 == 0:#每训练十组，就测试一组
                j, data = next(enumerate(testdataloader, 0))#返回testdataloader的下一个项目
                points,target,target_2=data
                points, target ,target_2 = points.cuda(), target.cuda(),target_2.cuda()
                points = torch.split(points,5,dim=0)
                oppoo1 = points[0]
                oppoo2 = points[1]
                oppoo3 = points[2]
                oppoo4 = points[3]
                oppoo1 =oppoo1.view(1,5,12,5)
                oppoo2 =oppoo2.view(1,5,12,5)
                oppoo3 =oppoo3.view(1,5,12,5)
                oppoo4 =oppoo4.view(1,5,12,5)
                points = torch.cat([oppoo1,oppoo2,oppoo3,oppoo4],dim=0)
                #points = points.transpose(2, 1)
                for name, module in classifier.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.training = False
                optimizer.zero_grad()
                optimizer2.zero_grad()
                pred= classifier(points,h0,c0,h0,c0)
                predd= classifier2(points,h0,c0,h0,c0)
                pred = pred.view(-1, num_classes)
                predd = predd.view(-1, 2)
                target = target.view(-1, 1)[:, 0] - 1
                target_2 = target_2.view(-1, 1)[:, 0] - 1
                # target = target.view(240, 1)
                loss2 = F.nll_loss(predd, target_2)
                loss = F.nll_loss(pred, target)
                c_1 = loss.item()/(loss2.item()+loss.item())
                loss3=(1-c_1)*loss+c_1*loss2
                loss3.backward()
                optimizer.step()
                optimizer2.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss3.item(), correct.item()/float(batchSize *sequenceSize* npoints)))
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(accADD/211)

        tensor21 = torch.cat((tensor21,tensor1),0)
        tensor22 = torch.cat((tensor22,tensor2),0)
        tensor1 = torch.zeros(0).cuda()
        tensor2 = torch.zeros(0).cuda()


        writer.add_scalar('Acc', accADD/211, epoch)
        writer.add_scalar('TrainLoss', loss.item(), epoch)
        # modell = SetTransformer(npoints, num_classes, feature_transform=feature_transform)
        # modell.train()
        # ppp=torch.rand(20,5,npoints)
        # with SummaryWriter('./SetTransformer') as w:
        #     w.add_graph(modell, (ppp))
        
        writer.close()

        torch.save(classifier.state_dict(), '%s/seg_Model_mmMesh_62_%d.pth' % (outf, epoch))

    x,y = Cor_Pred(tensor21,tensor22,nepoch)
    ddx = pd.DataFrame(x)
    ddy = pd.DataFrame(y)
    ddx.to_csv('./mmMesh_62_cor_pred.csv') 
    ddy.to_csv('./mmMesh_62_wor_pred.csv')





    ##模型训练完之后，最后分割一遍，测试效果
    testnum=0
    accurate=0
    shape_ious = []
    shape_mAcc = []
    shape_mAcc2 = []
    shape_ious2 = []
    for i,data in enumerate(tqdm(testdataloader, 0)):
        points,target,target0=data
        if(points.size()[0]<batchSize*sequenceSize):
                continue
        if i % 10 == 0:#每训练十组，就测试一组
            j, data = next(enumerate(testdataloader, 0))#返回testdataloader的下一个项目
            points,target,waste=data
            points, target = points.cuda(), target.cuda()

        points, target = points.cuda(), target.cuda() 
        points = torch.split(points,5,dim=0)
        oppooo1 = points[0]
        oppooo2 = points[1]
        oppooo3 = points[2]
        oppooo4 = points[3]
        oppooo1 =oppooo1.view(1,5,12,5)
        oppooo2 =oppooo2.view(1,5,12,5)
        oppooo3 =oppooo3.view(1,5,12,5)
        oppooo4 =oppooo4.view(1,5,12,5)
        points = torch.cat([oppooo1,oppooo2,oppooo3,oppooo4],dim=0)             
        classifier = classifier.eval()
        pred= classifier(points,h0,c0,h0,c0)
        pred_choice = pred.data.max(1)[1]

        pred_np = pred_choice.cpu().data.numpy()
        pred_np = pred_np.reshape(-1,npoints)
        target_np = target.cpu().data.numpy() - 1
       
       

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)#np.unique(target_np[shape_idx])
            part_ious = []
            part_mAcc = []
            part_mAcc2 = []
            part_ious2 = []
            for part in parts:
                a=np.where(target_np[shape_idx] == part)
                aa=a[0]
                aaa=aa.shape
                aaaa=aaa[0]
                d=np.where(pred_np[shape_idx] == part)
                dd=d[0]
                ddd=dd.shape
                dddd=ddd[0]
                M = np.sum(np.where(target_np[shape_idx] == part))
                b=np.where((pred_np[shape_idx] == part) & (target_np[shape_idx] == part))
                bb=b[0]
                bbb=bb.shape
                bbbb=bbb[0]
                A = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                c=np.where((pred_np[shape_idx] == part) | (target_np[shape_idx] == part))
                cc=c[0]
                ccc=cc.shape
                cccc=ccc[0]
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if M == 0:
                    classACC = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    classACC = A/M
                if aaaa == 0:
                    classACC2 = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    classACC2 = bbbb/aaaa
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                if cccc == 0:
                    iou2 = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou2 = bbbb/cccc
                part_ious.append(iou)
                part_ious2.append(iou2)
                part_mAcc.append(classACC)
                part_mAcc2.append(classACC2)
            shape_ious.append(np.mean(part_ious))
            shape_ious2.append(np.mean(part_ious2))
            shape_mAcc.append(np.mean(part_mAcc))
            shape_mAcc2.append(np.mean(part_mAcc2))

        print('')
        pred_np=pred_np.ravel()
        target_np=target_np.ravel()
        for i in range(pred_np.shape[0]):
            if(pred_np[i]==target_np[i]):
                accurate+=1
        testnum+=pred_np.shape[0]

    print('accuracy rate : {}'.format(accurate/float(testnum)))

    print('mAcc: {}'.format(np.mean(shape_mAcc)))

    print('mAcc2: {}'.format(np.mean(shape_mAcc2)))

    print("mIOU : {}".format( np.mean(shape_ious)))

    print("mIOU2 : {}".format( np.mean(shape_ious2)))
    