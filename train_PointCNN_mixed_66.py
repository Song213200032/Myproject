from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from Model_PoinCNN import *
from loaddata import ShapeNetDataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd   
from scipy.io import loadmat 
from torch.utils.tensorboard import SummaryWriter
writer =SummaryWriter('./PointCNN_mixed66')


manualSeed = random.randint(1, 10000)  # fix seed

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

in_channels = 5
out_channels = 24
dims = 5
K = 5
P = 12
C_mid = 32
depth_multiplier = 4

batchSize=4 #一批处理四个帧
sequenceSize=5 #一帧
workers=0
outf='seg'
feature_transform=True
Model_PoinCNN=''
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
AbbPointCNN = lambda a, b, c, d, e: PointCNN(a, b, 5, c, d, e, knn_indices_func_gpu)# c*d need to be smaller than num_points
# a  C_in: Input dimension of the points' features.
# b  C_out: Output dimension of the representative point features.
# c  K: Number of neighbors to convolve over.
# d  D: "Spread" of neighboring points.
# e  P: Number of representative points.
class Classifier(nn.Module):

        def __init__(self):
            super(Classifier, self).__init__()

            self.pcnn1 = AbbPointCNN(5, 32, 5, 1, 12)
            self.pcnn2 = AbbPointCNN(32, 64, 5, 2, 12)
            self.pcnn3 = AbbPointCNN(64, 96, 5, 2, 12)
            self.pcnn4 = AbbPointCNN(96, 128, 5, 2, 12)
            self.pcnn5 = AbbPointCNN(133, 160, 5, 2, 12)
            self.pcnn6 = AbbPointCNN(160, 128, 5, 2, 12)
            self.pcnn7 = AbbPointCNN(128, 96, 5, 2, 12)
            self.pcnn8 = AbbPointCNN(96, 64, 5, 2, 12)
            self.pcnn9 = AbbPointCNN(64, 6, 5, 1, 12)


            self.fcn = nn.Sequential(
               Dense(32, num_classes, with_bn=False, activation=None)
            )

        def forward(self, x):
            _,rep_pts,rep_fts = x
            x = self.pcnn1(x)
            if False:
               print("Making graph...")
               k = make_dot(x[1])

               print("Viewing...")
               k.view()
               print("DONE")

               assert False
            x = self.pcnn2((rep_pts,rep_pts,x))
            x = self.pcnn3((rep_pts,rep_pts,x))
            x = self.pcnn4((rep_pts,rep_pts,x))

            n_pts = x.shape[1]
            pc_feat = x.max(1,keepdim=True)[0]
            pc_feat=pc_feat.repeat(1,n_pts,1)
            x = torch.cat([rep_pts,pc_feat], 2)


            x = self.pcnn5((rep_pts,rep_pts,x))
            x = self.pcnn6((rep_pts,rep_pts,x))
            x = self.pcnn7((rep_pts,rep_pts,x))
            x = self.pcnn8((rep_pts,rep_pts,x))
            x = self.pcnn9((rep_pts,rep_pts,x))# grab features

            # logits = self.fcn(x)
            logits = x
            
            return logits
   

print("------Building model-------")
classifier = Classifier().cuda()
classifier2 = Classifier().cuda()
print("------Successfully Built model-------")

if Model_PoinCNN != '':
    classifier.load_state_dict(torch.load(Model_PoinCNN))

optimizer = optim.Adam(classifier.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
optimizer2 = optim.Adam(classifier2.parameters(), lr=0.01, betas=(0.9, 0.999))
scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=20, gamma=0.5)

classifier.cuda()
classifier2.cuda()

num_batch = len(dataset) / (batchSize*sequenceSize)
#num_batch代表的是，在一轮训练中，共有多少批次数据需要处理
#batchSize*sequenceSize代表的是，一批数据中有多少个帧点云

if __name__=='__main__':
    tensor1 = torch.zeros(0).cuda()
    tensor2 = torch.zeros(0).cuda()
    tensor21 = torch.zeros(0).cuda()
    tensor22 = torch.zeros(0).cuda()
    for epoch in range(nepoch):
        scheduler.step() #更新optimizer中的学习率
        scheduler2.step() #更新optimizer中的学习率
        accADD=0
        for i, data in enumerate(dataloader, 0): #enumerate函数先返回下标i，后返回元素值data
            points,target,waste=data #data中包含两个变量，分别是point_set 以及seg(标签数据)#points=[20 batch ,24 num_points ,5 num_dims]; target=[20 batch , 24 num_points];
            if(points.size()[0]<batchSize*sequenceSize):#batchSize:4    seqenceSize:5
                continue
            points, target = points.cuda(), target.cuda()#将CPU上的Tensor或变量放到GPU上
            optimizer.zero_grad()
            optimizer2.zero_grad()
            classifier.train()
            classifier2.train()
            rep_fts = torch.zeros(20,P,dims)
            rep_fts = rep_fts.cuda()
            pred= classifier((points,points,rep_fts))#pred[20,12,6]
            predd= classifier2((points,points,rep_fts))#pred[20,12,6]
            #pred - g_loc;   att - attn_weights;
            # h - hn;         c  -  cn;
            #可以把hn理解为当前时刻，LSTM层的输出结果，而cn是记忆单元中的值
            #g_vec则是包括当前时刻以及之前时刻所有hn的输出值
            #g_loc是经过全连接层的g_vec，分了下类，提取了一下特征
            #print(pred.size())   #pred[20,12,6]
            #print(target.size())
            pred = pred.view(-1, num_classes) #重构张量的维度（x，6）
            predd = predd.view(-1, num_classes) #重构张量的维度（x，6）
            #经过打印发现.view()只是将不同的矩阵从三维按顺序拼接成了二维的形状
            target = target.view(-1, 1)[:, 0] - 1
            loss = F.nll_loss(pred, target)
            loss2 = F.nll_loss(predd, target)
            c_1 = loss.item()/(loss2.item()+loss.item())
            loss3=(1-c_1)*loss+c_1*loss2
            loss3.backward()
            optimizer.step()
            optimizer2.step()
            #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            #print(pred.data)
            #print(pred.data.shape)
            pred_choice = pred.data.max(1)[1]
            tensor1 = torch.cat((tensor1,pred_choice),0)
            tensor2 = torch.cat((tensor2,target),0)
            #torch.max(1)[1]， 只返回矩阵每一行最大值的每个索引
            #print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            #print(pred_choice)
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
                points, target ,waste = data
                points, target = points.cuda(), target.cuda()
                for name, module in classifier.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        module.training = False
                rep_fts2 = torch.zeros(20,P,dims)
                rep_fts2 = rep_fts2.cuda()
                optimizer.zero_grad()
                optimizer2.zero_grad()
                pred = classifier((points,points,rep_fts2))
                predd = classifier2((points,points,rep_fts2))
                pred = pred.view(-1, num_classes)
                predd = predd.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                loss2 = F.nll_loss(predd, target)
                c_1 = loss.item()/(loss2.item()+loss.item())
                loss3=(1-c_1)*loss+c_1*loss2
                loss3.backward()
                optimizer.step()
                optimizer2.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(batchSize *sequenceSize* npoints)))
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print(accADD/211)


        tensor21 = torch.cat((tensor21,tensor1),0)
        tensor22 = torch.cat((tensor22,tensor2),0)
        tensor1 = torch.zeros(0).cuda()
        tensor2 = torch.zeros(0).cuda()


        writer.add_scalar('Acc', accADD/211, epoch)
        writer.add_scalar('TrainLoss', loss.item(), epoch)
        # modell = Classifier().cuda()
        # modell.train()                                          
        # ppp=torch.rand(20,npoints,5)
        # rep_ftss = torch.zeros(20,P,dims)
        # rep_ftss = rep_ftss.cuda()
        # with SummaryWriter('./EdgeNet') as w:
        #     w.add_graph(modell, (ppp,ppp,rep_ftss))
        
        writer.close()

        torch.save(classifier.state_dict(), '%s/seg_Model_PoinCNN_66_%d.pth' % (outf, epoch))


    x,y = Cor_Pred(tensor21,tensor22,nepoch)
    ddx = pd.DataFrame(x)
    ddy = pd.DataFrame(y)
    ddx.to_csv('./PointCNN_66_cor_pred.csv') 
    ddy.to_csv('./PointCNN_66_wor_pred.csv')



    
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
        classifier = classifier.eval()
        rep_fts3 = torch.zeros(20,P,dims)
        rep_fts3 = rep_fts3.cuda()
        pred= classifier((points,points,rep_fts3))
        pred_choice = pred.data.max(2)[1]

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