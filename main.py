import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
# 生成两个示例数组
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([5, 4, 3, 2, 1])

# # 计算相关性指标
# cc = np.corrcoef(x, y)[0, 1]

# print("cc相关性指标为:", cc)

y = io.loadmat('D:/PyCharmProject/data_TSK/class of data/label.mat')['label']
y = np.array(y)
print(y.shape)
n,_ = y.shape

x = [i for i in range(1, n+1)]
x=np.array(x)
print(x.shape)

plt.plot(x[:360],y[:360])

plt.show()