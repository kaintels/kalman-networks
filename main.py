import torch
import torch.nn as nn
import numpy as np
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
import random
import kalman
random_seed = 777


torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

A = 2
H = 1.5
Q = 1
R = 9

x_0 = 1
P_0 = 1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.encoder = nn.Linear(1000, 300)
        self.decoder = nn.Linear(300, 1000)
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode

ecg = nk.ecg_simulate(duration=1, heart_rate=80, random_state=random_seed, noise=0.0)

len_sample = len(ecg)
rn = 50;
ecg_noise = torch.FloatTensor(ecg) + torch.randn(len_sample) / rn;

learn_rate = 0.01
n_epochs = 150

model = Net()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(lr = learn_rate, params=model.parameters())

losses = []
for i in range(0, n_epochs):
    
    esti_save = torch.zeros(len_sample)
    for u in range(len_sample):
        if u == 0:
            x_esti, P = x_0, P_0
        else:
            x_esti, P = kalman.kalman_filter(ecg_noise[u], A, H, Q, R, P, x_esti)
        esti_save[u] = torch.FloatTensor(np.array([x_esti]))

    optimizer.zero_grad()
    y_pred = model(ecg_noise)
    loss = mse_loss(y_pred, esti_save)
    loss.backward()
    optimizer.step()

    
    print(loss.item())
    losses.append(loss.item())

# plt.plot(losses)
# plt.show()

# model2 = Net()
# mse_loss2 = nn.MSELoss()
# optimizer2 = torch.optim.Adam(lr = learn_rate, params=model2.parameters())

# losses = []
# for i in range(0, n_epochs):
    
#     esti_save = torch.zeros(len_sample)
#     for u in range(len_sample):
#         if u == 0:
#             x_esti, P = x_0, P_0
#         else:
#             x_esti, P = kalman_filter(ecg_noise[u], x_esti, P)
#         esti_save[u] = x_esti

#     optimizer2.zero_grad()
#     y_pred2 = model2(ecg_noise)
#     loss2 = mse_loss(y_pred2, ecg_noise)
#     loss2.backward()
#     optimizer2.step()
#     print(loss2.item())
#     losses.append(loss2.item())

# plt.plot(losses)
# plt.show()

# plt.plot(ecg_noise, 'r', alpha = 0.7)
# plt.plot(y_pred2.detach(), 'b', alpha = 0.3)
# plt.plot(y_pred.detach(), 'k')
# plt.show()