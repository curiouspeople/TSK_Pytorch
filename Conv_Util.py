import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out




class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride=4, dilation=2, padding=0.5, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.3):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        out = self.network(x)
        # print("outshape: ",out.shape)
        return out


# pytorch封装卷积层  空洞卷积
class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        # 定义六层卷积层
        # 两层HDC（1,2,5,1,2,5）
        self.conv = nn.Sequential(
            # 第一层 (3-1)*1+1=3 （64-3)/1 + 1 =62
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第二层 (3-1)*2+1=5 （62-5)/1 + 1 =58
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(32),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第三层 (3-1)*5+1=11  (58-11)/1 +1=48
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=5),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第四层(3-1)*1+1=3 （48-3)/1 + 1 =46
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=1),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第五层 (3-1)*2+1=5 （46-5)/1 + 1 =42
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(64),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True),
            # 第六层 (3-1)*5+1=11  (42-11)/1 +1=32
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0, dilation=5),
            nn.BatchNorm2d(128),
            # inplace-选择是否进行覆盖运算
            nn.ReLU(inplace=True)

        )
        # 输出层,将通道数变为分类数量
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 图片经过三层卷积，输出维度变为(batch_size,C_out,H,W)
        out = self.conv(x)
        # 使用平均池化层将图片的大小变为1x1,第二个参数为最后输出的长和宽（这里默认相等了）
        out = F.avg_pool2d(out, 32)
        # 将张量out从shape batchx128x1x1 变为 batch x128
        out = out.squeeze()
        # 输入到全连接层将输出的维度变为3
        out = self.fc(out)
        return out


# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        print(x.shape)
        x = self.avg_pool(x)
        print(x.shape)
        x = self.fc1(x)
        x = self.relu1(x)
        avg_out = self.fc2(x)
        # avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

# 深度可分离卷积
# 用于增加通道数，便于馈入到通道注意力当中
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        # [885, 4, 17 ,25] -> [885, 32, 17, 25]
        output_data = self.pointwise_conv(self.depthwise_conv(x))
        return output_data


import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        #头数n_head，模型维度d_model，查询向量维度d_k，数值向量维度d_v，以及dropout概率。
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_x, -1)
        q = self.dropout(self.fc(q))

        q += residual
        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q x k^T
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            # 把mask中为0的数置为-1e9, 用于decoder中的masked self-attention
            attn = attn.masked_fill(mask == 0, -1e9)

        # dim=-1表示对最后一维softmax
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn