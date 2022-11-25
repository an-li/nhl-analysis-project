import torch.nn as nn
import torch.nn.functional as F


class NetSGD(nn.Module):

    def __init__(self):
        super(NetSGD, self).__init__()
        self.fc1 = nn.Linear(35, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.dropout(x, p=0.1)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.sigmoid(x)

        return x
