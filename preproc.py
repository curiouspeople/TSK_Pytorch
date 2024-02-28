import torch
import torch.nn as nn
import numpy as np
import scipy.io as io
from Conv_Util import TemporalConvNet,DepthWiseConv,cbam_block, DepthwiseSeparableConv
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
#数据输入维度是[3600, 31, 1000]，表示时间，通道和采样数据
class Conv_Pro(nn.Module):
    def __init__(self):
        super(Conv_Pro, self).__init__()
        self.TCN = TemporalConvNet(31,[25,25,25,4])
        # self.dethConv = DepthwiseSeparableConv(1, 4)
        # self.cbam = cbam_block(4)


    def forward(self,x):
        print(x.shape)
        temp = self.TCN(x)
        out = temp.reshape([3600,-1])
        # out = self.dethConv(temp)
        # print(temp.shape)
        # out = self.cbam(temp)
        return out




# 3600,31
if __name__ == '__main__':
    X= torch.rand((3600,31,1000))
    # X = io.loadmat('E:/LSH/mydata/data_process/cg_renwu1_1.mat')['data1']# [31,3600000]
    # X=np.array(X)[:, :-1]
    X=X.reshape((31,3600,-1))
    print(X.shape)
    X=np.swapaxes(X,0,1)
    # X =torch.Tensor(X)
    tcn = Conv_Pro()
    # print(tcn(X).shape)
    arr = tcn(X)
    # arr=arr.detach().numpy()
    # arr = np.array(arr)
    print(arr.shape)
    # np.save('data_pro.npy',arr)
