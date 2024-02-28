import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#注意力层-> Pengrui Li
class Attention_Layer(nn.Module):

    # 用来实现mask-attention layer
    def __init__(self, hidden_dim):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim

        # 单向的RNN
        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs):

        # size = inputs.size()
        # 计算生成QKV矩阵
        Q = self.Q_linear(inputs)
        K = self.K_linear(inputs).permute(1,0)
        # K = self.K_linear(inputs).permute(0, 2, 1)  # 先进行一次转置
        V = self.V_linear(inputs)

        # 下面开始计算
        alpha = torch.matmul(Q, K)
        alpha = F.softmax(alpha, dim=1)
        out = torch.matmul(alpha, V)

        return out
# 定义自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        print(attended_values.size())
        return attended_values

# 定义自注意力分类器模型
class SelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(SelfAttentionClassifier, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        attended_values = self.attention(x)
        x = attended_values.mean(dim=1)  # 对每个位置的向量求平均
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
#通道注意力
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

#多头注意力
# 定义多头自注意力模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # 将输入向量拆分为多个头
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # 注意力加权求和
        attended_values = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 经过线性变换和残差连接
        x = self.fc(attended_values) + x

        return x
#空间注意力 CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(30)
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, _, _ = x.size()
        b,c = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = self.avg_pool(x)
        y = self.fc(y).view(b, c)
        return x * self.sigmoid(y)