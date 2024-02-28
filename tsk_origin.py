import numpy as np
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK

time_start = time.time()
# Prepare dataset
import scipy.io as io
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)  # X: [n_samples, n_features], y: [n_samples, 1]
# n_class = len(np.unique(y))  # Num. of class
X = io.loadmat('D:/PyCharmProject/data_TSK/class of data/data.mat')['X']
X[np.isnan(X)] = 0
y = io.loadmat('D:/PyCharmProject/data_TSK/class of data/label.mat')['label']

z=[]
for i in range(len(y)):
    if y[i] <= 0.35:
        z.append(0)
    else:
        z.append(1)
y=z

y=np.array(y)
X=np.array(X)
n_class=2


# split train-test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
    x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
))

# Z-score
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# Define TSK model parameters
n_rule = 30  # Num. of rules
lr = 0.01  # learning rate
weight_decay = 1e-8
consbn = False
order = 1

# --------- Define antecedent ------------
init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)
gmf = nn.Sequential(
    AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
    nn.LayerNorm(n_rule),
    nn.ReLU()
)
# set high_dim=True is highly recommended.

# --------- Define full TSK model ------------
model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None)

# ----------------- optimizer ----------------------------
ante_param, other_param = [], []
for n, p in model.named_parameters():
    if "center" in n or "sigma" in n:
        ante_param.append(p)
    else:
        other_param.append(p)
optimizer = AdamW(
    [{'params': ante_param, "weight_decay": 0},
    {'params': other_param, "weight_decay": weight_decay},],
    lr=lr
)
# ----------------- split 10% data for earlystopping -----------------
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
# ----------------- define the earlystopping callback -----------------
EACC = EarlyStoppingACC(x_val, y_val, verbose=1, patience=20, save_path="tmp_class.pkl")

wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
              epochs=300, callbacks=[EACC])
wrapper.fit(x_train, y_train)
wrapper.load("tmp_class.pkl")

y_pred = wrapper.predict(x_test).argmax(axis=1)
print("[TSK] ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))
time_end = time.time()
time_cost = time_end-time_start
print(time_cost)
