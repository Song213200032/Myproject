from __future__ import print_function
import xxlimited
from matplotlib import transforms
from sqlalchemy import false
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch import cuda, FloatTensor, LongTensor
from typing import Tuple, Callable, Optional
from typing import Union


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


# Types to allow for both CPU and GPU models.
UFloatTensor = Union[FloatTensor, cuda.FloatTensor]
ULongTensor = Union[LongTensor, cuda.LongTensor]

def knn_indices_func_gpu(rep_pts : cuda.FloatTensor,  # (N, pts, dim)
                         pts : cuda.FloatTensor,      # (N, x, dim)
                         k : int, d : int
                        ) -> cuda.LongTensor:         # (N, pts, K)
    """
    GPU-based Indexing function based on K-Nearest Neighbors search.
    Very memory intensive, and thus unoptimal for large numbers of points.
    :param rep_pts: Representative points.
    :param pts: Point cloud to get indices from.
    :param K: Number of nearest neighbors to collect.
    :param D: "Spread" of neighboring points.
    :return: Array of indices, P_idx, into pts such that pts[n][P_idx[n],:]
    is the set k-nearest neighbors for the representative points in pts[n].
    """
    region_idx = []

    for n, qry in enumerate(rep_pts):
        ref = pts[n]
        n, dd = ref.size()
        m, dd = qry.size()
        mref = ref.expand(m, n, dd)
        mqry = qry.expand(n, m, dd).transpose(0, 1)
        dist2 = torch.sum((mqry - mref)**2, 2).squeeze()
        _, inds = torch.topk(dist2, d*k+1, dim = 1, largest = False)
        region_idx.append(inds[:,1::d])

    region_idx = torch.stack(region_idx, dim = 0)
    return region_idx

class Dense(nn.Module):
    """
    Single layer perceptron with optional activation, batch normalization, and dropout.
    """

    def __init__(self, in_features : int, out_features : int,
                 drop_rate : int = 0, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                ) -> None:
        """
        :param in_features: Length of input featuers (last dimension).
        :param out_features: Length of output features (last dimension).
        :param drop_rate: Drop rate to be applied after activation.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        # self.bn = LayerNorm(out_channels) if with_bn else None
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Linear.
        :return: Tensor with linear layer and optional activation, batchnorm,
        and dropout applied.
        """
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        # if self.bn:
        #     x = self.bn(x)
        if self.drop:
            x = self.drop(x)
        return x

def EndChannels(f, make_contiguous = False):
    """ Class decorator to apply 2D convolution along end channels. """

    class WrappedLayer(nn.Module):

        def __init__(self):
            super(WrappedLayer, self).__init__()
            self.f = f

        def forward(self, x):
            x = x.permute(0,3,1,2)
            x = self.f(x)
            x = x.permute(0,2,3,1)
            return x

    return WrappedLayer()

class Conv(nn.Module):
    """
    2D convolutional layer with optional activation and batch normalization.
    """

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]], with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias = not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.9) if with_bn else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with convolutional layer and optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class SepConv(nn.Module):
    """ Depthwise separable convolution with optional activation and batch normalization"""

    def __init__(self, in_channels : int, out_channels : int,
                 kernel_size : Union[int, Tuple[int, int]],
                 depth_multiplier : int = 1, with_bn : bool = True,
                 activation : Callable[[UFloatTensor], UFloatTensor] = nn.ReLU()
                 ) -> None:
        """
        :param in_channels: Length of input featuers (first dimension).
        :param out_channels: Length of output features (first dimension).
        :param kernel_size: Size of convolutional kernel.
        :depth_multiplier: Depth multiplier for middle part of separable convolution.
        :param with_bn: Whether or not to apply batch normalization.
        :param activation: Activation function.
        """
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups = in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias = not with_bn)
        )

        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum = 0.9) if with_bn else None

    def forward(self, x : UFloatTensor) -> UFloatTensor:
        """
        :param x: Any input tensor that can be input into nn.Conv2d.
        :return: Tensor with depthwise separable convolutional layer and
        optional activation and batchnorm applied.
        """
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.bn:
            x = self.bn(x)
        return x

class XConv(nn.Module):
    """ Convolution over a single point and its neighbors.  """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int,
                 P : int, C_mid : int, depth_multiplier : int) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param P: Number of representative points.
        :param C_mid: Dimensionality of lifted point features.
        :param depth_multiplier: Depth multiplier for internal depthwise separable convolution.
        """
        super(XConv, self).__init__()

        if __debug__:
            # Only needed for assertions.
            self.C_in = C_in
            self.C_mid = C_mid
            self.dims = dims
            self.K = K

        self.P = P

        # Additional processing layers
        # self.pts_layernorm = LayerNorm(2, momentum = 0.9)

        # Main dense linear layers
        self.dense1 = Dense(dims, C_mid)
        self.dense2 = Dense(C_mid, C_mid)

        # Layers to generate X
        self.x_trans = nn.Sequential(
            EndChannels(Conv(
                in_channels = dims,
                out_channels = K*K,
                kernel_size = (1, K),
                with_bn = False
            )),
            Dense(K*K, K*K, with_bn = False),
            Dense(K*K, K*K, with_bn = False, activation = None)
        )
        
        self.end_conv = EndChannels(SepConv(
            in_channels = C_mid + C_in,
            out_channels = C_out,
            kernel_size = (1, K),
            depth_multiplier = depth_multiplier
        )).cuda()
        
    def forward(self, x : Tuple[UFloatTensor,            # (N, P, dims)
                                UFloatTensor,            # (N, P, K, dims)
                                Optional[UFloatTensor]]  # (N, P, K, C_in)
               ) -> UFloatTensor:                        # (N, K, C_out)
        """
        Applies XConv to the input data.
        :param x: (rep_pt, pts, fts) where
          - rep_pt: Representative point.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the feature
          associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated into point rep_pt.
        """
        rep_pt, pts, fts = x

        if fts is not None:
            assert(rep_pt.size()[0] == pts.size()[0] == fts.size()[0])  # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1] == fts.size()[1])  # Check P is equal.
            assert(pts.size()[2] == fts.size()[2] == self.K)            # Check K is equal.
            assert(fts.size()[3] == self.C_in)                          # Check C_in is equal.
        else:
            assert(rep_pt.size()[0] == pts.size()[0])                   # Check N is equal.
            assert(rep_pt.size()[1] == pts.size()[1])                   # Check P is equal.
            assert(pts.size()[2] == self.K)                             # Check K is equal.
        assert(rep_pt.size()[2] == pts.size()[3] == self.dims)          # Check dims is equal.

        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim = 2)  # (N, P, 1, dims)

        #-1 Move pts to local coordinate system of rep_pt.
        pts_local = pts - p_center  # (N, P, K, dims)
        # pts_local = self.pts_layernorm(pts - p_center)

        #-2 Individually lift each point into C_mid space.
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted  = self.dense2(fts_lifted0)  # (N, P, K, C_mid)

        if fts is None:
            fts_cat = fts_lifted
        else:
            #-3 Concaternate lifted features and features, Get concaternated features 
            fts_cat = torch.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)

        #-4 Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, P, self.K, self.K)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)

        #-5 Weight and permute fts_cat with the learned X.
        fts_X = torch.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X)
        fts_p = fts_p.squeeze(dim = 2)
        return fts_p

class PointCNN(nn.Module):
    """ Pointwise convolutional model. """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                                            UFloatTensor,  # (N, x, dims)
                                            int, int],
                                           ULongTensor]    # (N, P, K)
                ) -> None:
        """
        :param C_in: Input dimension of the points' features.
        :param C_out: Output dimension of the representative point features.
        :param dims: Spatial dimensionality of points.
        :param K: Number of neighbors to convolve over.
        :param D: "Spread" of neighboring points.
        :param P: Number of representative points.
        :param r_indices_func: Selector function of the type,
          INPUTS
          rep_pts : Representative points.
          pts  : Point cloud.
          K : Number of points for each region.
          D : "Spread" of neighboring points.

          OUTPUT
          pts_idx : Array of indices into pts such that pts[pts_idx] is the set
          of points in the "region" around rep_pt.
        """
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4

        if C_in == 0:
            depth_multiplier = 1
        else:
            depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)

        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D

    def select_region(self, pts : UFloatTensor,  # (N, x, dims)
                      pts_idx : ULongTensor      # (N, P, K)
                     ) -> UFloatTensor:          # (P, K, dims)
        """
        Selects neighborhood points based on output of r_indices_func.
        :param pts: Point cloud to select regional points from.
        :param pts_idx: Indices of points in region to be selected.
        :return: Local neighborhoods around each representative point.
        """
        regions = torch.stack([
            pts[n][idx,:] for n, idx in enumerate(torch.unbind(pts_idx, dim = 0))
        ], dim = 0)
        return regions

    def forward(self, x : Tuple[FloatTensor,  # (N, P, dims)
                                FloatTensor,  # (N, x, dims)
                                FloatTensor]  # (N, x, C_in)
               ) -> FloatTensor:              # (N, P, C_out)
        """
        Given a set of representative points, a point cloud, and its
        corresponding features, return a new set of representative points with
        features projected from the point cloud.
        :param x: (rep_pts, pts, fts) where
          - rep_pts: Representative points.
          - pts: Regional point cloud such that fts[:,p_idx,:] is the
          feature associated with pts[:,p_idx,:].
          - fts: Regional features such that pts[:,p_idx,:] is the feature
          associated with fts[:,p_idx,:].
        :return: Features aggregated to rep_pts.
        """
        rep_pts, pts, fts = x
        fts = self.dense(fts) if fts is not None else fts

        # This step takes ~97% of the time. Prime target for optimization: KNN on GPU.
        pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu()).cuda()
        # -------------------------------------------------------------------------- #

        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))

        return fts_p


class RandPointCNN(nn.Module):
    """ PointCNN with randomly subsampled representative points. """

    def __init__(self, C_in : int, C_out : int, dims : int, K : int, D : int, P : int,
                 r_indices_func : Callable[[UFloatTensor,  # (N, P, dims)
                                            UFloatTensor,  # (N, x, dims)
                                            int, int],
                                           ULongTensor]    # (N, P, K)
                ) -> None:
        """ See documentation for PointCNN. """
        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P

    def forward(self, x : Tuple[UFloatTensor,  # (N, x, dims)
                                UFloatTensor]  # (N, x, dims)
               ) -> Tuple[UFloatTensor,        # (N, P, dims)
                          UFloatTensor]:       # (N, P, C_out)
        """
        Given a point cloud, and its corresponding features, return a new set
        of randomly-sampled representative points with features projected from
        the point cloud.
        :param x: (pts, fts) where
         - pts: Regional point cloud such that fts[:,p_idx,:] is the
        feature associated with pts[:,p_idx,:].
         - fts: Regional features such that pts[:,p_idx,:] is the feature
        associated with fts[:,p_idx,:].
        :return: Randomly subsampled points and their features.
        """
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            idx = np.random.choice(pts.size()[1], self.P, replace = False).tolist()
            rep_pts = pts[:,idx,:]
        else:
            # All input points are representative points.
            rep_pts = pts
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        return rep_pts, rep_pts_fts


if __name__=='__main__':
    
    num_classes=6
    in_channels = 5
    out_channels = 24
    dims = 5
    K = 5
    P = 12
    C_mid = 32
    depth_multiplier = 4
    rep_pts = torch.rand(20,P,dims)
    rep_pts = rep_pts.cuda()
    pts_regional = torch.rand(20,P,K,dims)
    pts_regional = pts_regional.cuda()
    fts_regional = torch.rand(20,P,K,in_channels)
    fts_regional = fts_regional.cuda()
    x_conv = XConv(in_channels, out_channels, dims, K, P, C_mid, depth_multiplier)
    x_conv.cuda()
    fts_p = x_conv((rep_pts, pts_regional, fts_regional))
    print(fts_p.size())
    AbbPointCNN = lambda a, b, c, d, e: PointCNN(a, b, 5, c, d, e, knn_indices_func_gpu)# c*d need to be smaller than num_points
    # a  C_in: Input dimension of the points' features.
    # b  C_out: Output dimension of the representative point features.
    # c  K: Number of neighbors to convolve over.
    # d  D: "Spread" of neighboring points.
    # e  P: Number of representative points.
    class Classifier(nn.Module):

        def __init__(self):
            super(Classifier, self).__init__()

            self.pcnn1 = AbbPointCNN(5, 64, 5, 1, 12)
            self.pcnn2 = AbbPointCNN(64, 64, 5, 2, 12)
            self.pcnn3 = AbbPointCNN(64, 128, 5, 2, 12)
            self.pcnn4 = AbbPointCNN(128, 1024, 5, 2, 12)
            self.pcnn5 = AbbPointCNN(1088, 512, 5, 2, 12)
            self.pcnn6 = AbbPointCNN(512, 256, 5, 2, 12)
            self.pcnn7 = AbbPointCNN(256, 128, 5, 2, 12)
            self.pcnn8 = AbbPointCNN(128, 32, 5, 1, 12)

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
            x = self.pcnn4((rep_pts,rep_pts,x))#[20 12 1024]
            x = torch.cat([rep_pts, x], 1)#[20 12 1088]
            x = self.pcnn5((rep_pts,rep_pts,x))
            x = self.pcnn6((rep_pts,rep_pts,x))
            x = self.pcnn7((rep_pts,rep_pts,x))
            x = self.pcnn8((rep_pts,rep_pts,x))  # grab features

            logits = self.fcn(x)
            return logits
    classifier = Classifier().cuda()
    rep_fts = torch.zeros(20,P,dims)
    rep_fts = rep_fts.cuda()
    out = classifier((rep_pts,rep_pts,rep_fts))
    print(out.size())
    print(out)
