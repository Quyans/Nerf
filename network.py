import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(60, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(316, 256),
            nn.ReLU()
        )
        self.fc7 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc8 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc9 = nn.Linear(256, 256)
        self.addiSig = nn.Linear(256, 1)

        self.fc10 = nn.Sequential(
            nn.Linear(280, 128),
            nn.ReLU()
        )
        self.fc11 = nn.Sequential(
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

    def forward(self, x, dire):
        """
        前馈神经网络
        :param x:位置，一个tensor
        :param dire: 方向
        :return: sigma: 体积强度，一个值
                 radiance: tensor RGB
        """
        fc1_out = self.fc1(x)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)
        fc4_out = self.fc4(fc3_out)
        fc5_out = self.fc5(fc4_out)
        fc6_out = self.fc6(torch.cat((x, fc5_out),0))
        fc7_out = self.fc7(fc6_out)
        fc8_out = self.fc8(fc7_out)
        fc9_out = self.fc9(fc8_out)
        fc10_out = self.fc10(torch.cat((fc9_out, dire),0))
        sigma = self.addiSig(fc8_out)
        out = self.fc11(fc10_out)

        return sigma, out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

x = torch.rand(60)
dire = torch.rand(24)
y = torch.rand(3)
for i in range(500):
    outSig, outRGB = net.forward(x, dire)
    loss = loss_function(outRGB, y)
    print ("sig is %f, RGB is " % outSig, outRGB)
    loss.backward()
    optimizer.step()

print(outRGB)
