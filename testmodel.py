from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from model5 import *
from loaddata import ShapeNetDataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat 


# parser = argparse.ArgumentParser()
# parser.add_argument(
#     '--batchSize', type=int, default=32, help='input batch size')
# parser.add_argument(
#     '--workers', type=int, help='number of data loading workers', default=4)
# parser.add_argument(
#     '--nepoch', type=int, default=25, help='number of epochs to train for')
# parser.add_argument('--outf', type=str, default='seg', help='output folder')
# parser.add_argument('--model', type=str, default='', help='model path')
# parser.add_argument('--dataset', type=str, required=True, help="dataset path")
# parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
# parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

#opt = parser.parse_args()
#print(opt)

# opt.manualSeed = random.randint(1, 10000)  # fix seed
manualSeed = random.randint(1, 10000)  # fix seed

print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

batchSize=32
workers=2
outf='seg'
feature_transform=False
model=''
nepoch=25
npoints=12

path='labeledData'
pthpath='seg\\seg_model_24.pth'

dataset = ShapeNetDataset(root=path,split='train')
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchSize,
    shuffle=True,
    num_workers=int(workers))

test_dataset = ShapeNetDataset(root=path,split='test')
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batchSize,
    shuffle=True,
    num_workers=int(workers))

num_classes = 3
print('classes', num_classes)
try:
    os.makedirs(outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = GlobalTiModule(k=num_classes, feature_transform=feature_transform)
classifier.load_state_dict(torch.load(pthpath))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

classifier.cuda()

testnum=0
accurate=0
if __name__=='__main__':
    ## benchmark mIOU
    shape_ious = []
    for i,data in tqdm(enumerate(testdataloader, 0)):
        points,target=data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1
       
       

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)#np.unique(target_np[shape_idx])
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

        
        pred_np=pred_np.ravel()
        target_np=target_np.ravel()
        for i in range(pred_np.shape[0]):
            if(pred_np[i]==target_np[i]):
                accurate+=1
        testnum+=pred_np.shape[0]
    print('accuracy rate : {}'.format(accurate/float(testnum)))
    print("mIOU : {}".format( np.mean(shape_ious)))

