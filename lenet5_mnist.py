import torch
from torch import nn
from torch.nn import functional as F

class Lenet5(nn.Module):
    # for mnist dataset
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit = nn.Sequential(nn.Conv2d(1, 12, 5),
                                       nn.MaxPool2d(2),
                                       nn.Conv2d(12, 64, 5),
                                       nn.MaxPool2d(2),
                                       )
        self.fc_unit = nn.Sequential(nn.Linear(64*4*4, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 10),
                                     nn.ReLU()
                                     )
        # self.criterion = nn.CrossEntropyLoss()   # Softmax函数的数据不稳定，因此不直接写Softmax，
                                                   #我们把它包含在Cross Entropy中
                                                   # 分类问题使用交叉熵
        # x = torch.randn(2, 1, 28, 28)
        # out = self.conv_unit(x)
        # print('conv_out:', out.size())

    def forward(self, data):

        data = self.conv_unit(data)
        data = data.view(data.shape[0], 1024)
        logits = self.fc_unit(data)    # Softmax前面的部分称为logits

        #[b, 10]
        # pred = F.softmax(logits, dim=1)  # 在10上做softmax，因此dim=1
        # loss = self.criterion(logits, targets)   #(pred, targets)

        return logits

def main():

    net = Lenet5()
    tmp = torch.randn(2, 1, 28, 28)
    out = net(tmp)
    print('lenet out:', out.shape)


if __name__ == '__main__':
    main()