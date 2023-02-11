import torch
import torch.nn as nn

class KalmanNetV1(nn.Module):
    def __init__(self, x_0, p_0, A, H, Q, R):
        super(KalmanNetV1, self).__init__()

        self.x = nn.Parameter(torch.tensor(x_0, dtype=torch.float64))
        self.p = nn.Parameter(torch.tensor(p_0, dtype=torch.float64))

        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

    def forward(self, x):
        len_sample = x.shape[0]
        esti_save = torch.zeros(len_sample)

        for u in range(len_sample):
            x_pred = self.A * self.x
            P_pred = self.A * self.p * self.A + self.Q
            K = P_pred * self.H / (self.H * P_pred * self.H + self.R)
            with torch.no_grad():
                self.x.copy_(torch.tensor((x_pred + K * (x[u] - self.H * x_pred)).item()))
                self.p.copy_(torch.tensor((P_pred - K * self.H * P_pred).item()))
            esti_save[u] = self.x.item()

        return esti_save.reshape(len_sample, -1).requires_grad_(True)

if __name__ == "__main__":
    model = KalmanNetV1(x_0=1,p_0=1,A=2,H=1.5,Q=1,R=9)