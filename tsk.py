import numpy as np
import torch.cuda
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingMSE
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK

# read .mat file
import scipy.io as io
from preproc import rmse,Conv_Pro
import time

# 把原始数据进行处理，使用深度学习模型
#开始运行
time_start = time.time()  # 开始计时

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


# X = io.loadmat('E:/LSH/mydata/data_process/cg_renwu1_1.mat')['data1']
# X=np.array(X)[:, :-1]
# # print(X.shape)
# X=X.reshape((31,3600,-1))
# X=np.swapaxes(X,0,1)
# X =torch.Tensor(X)
X = io.loadmat('D:/PyCharmProject/data_TSK/class of data/data.mat')['X']
# X = np.load('data_pro.npy')
X[np.isnan(X)] = 0
y = io.loadmat('D:/PyCharmProject/data_TSK/class of data/label.mat')['label']
# y=y[:3600]#each sample labels
y=np.array(y)
X=np.array(X)
# split train-test
n_class=1
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
    x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
))

# Z-score
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# Define TSK model parameters
n_rule = 300  # Num. of rules 30
lr = 0.01 # learning rate
weight_decay = 1e-8 # 1e-8
consbn = False
order = 1# 1： one-order TSK Fuzzy System

# --------- Define antecedent ------------
init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)

# tcn =Conv_Pro()
gmf = nn.Sequential(
    # tcn(X),
    AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=False, init_center=init_center),
    nn.Dropout(0.3)
)
# --------- Define full TSK model ------------
#precons is self_model output for example cnn
model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=nn.BatchNorm1d(X.shape[1]))
# ----------------- optimizer ----------------------------
ante_param, other_param = [], []
for n, p in model.named_parameters():
    if "center" in n or "sigma" in n:
        ante_param.append(p)
        # print("ante_param",p.shape)
    else:
        other_param.append(p)
        # print("other_param:",p.shape)
optimizer = AdamW(
    [{'params': ante_param, "weight_decay": 0},
    {'params': other_param, "weight_decay": weight_decay},],
    lr=lr
)
# ----------------- split 10% data for earlystopping -----------------
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
# ----------------- define the earlystopping callback -----------------

EMSE = EarlyStoppingMSE(x_val, y_val, verbose=1, patience=20, save_path="tmp_regression.pkl")
ur = 1.5 # must > 0
ur_tau=0.1  # a float number between 0 and 1
wrapper = Wrapper(
    model, optimizer=optimizer, label_type='r', criterion=nn.MSELoss(), epochs=300, callbacks=[EMSE], ur=ur, ur_tau=ur_tau,consbn=consbn,device=device
)
# wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.MSELoss(),
#               epochs=300,label_type='r',callbacks=[EMSE],device=device)
wrapper.fit(x_train, y_train)
print("training sahape: ",x_train.shape,y_train.shape)

wrapper.load("tmp_regression.pkl")
y_pred = wrapper.predict(x_test)


print("[TSK] Test RMSE: {:.4f}".format(rmse(y_test, y_pred)))
time_end = time.time()  # 结束计时

# 计算相关系数
cc = np.corrcoef(y_test.T, y_pred.T)[0, 1]

print("Correlation Coefficient:", cc)


time_c = time_end - time_start  # 运行所花时间
print('time cost', time_c, 's')
