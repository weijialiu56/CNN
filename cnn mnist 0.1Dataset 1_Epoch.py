import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler

from torch.autograd.profiler import profile, record_function, ProfilerActivity

from lenet5_mnist import Lenet5
from resnet_mnist import ResNet18

import matplotlib.pyplot as plt

def main():

    batch_size = 128
    data_path = '/data/mnist'
    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0,), (1,)),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                    ])

    mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_indices = torch.randperm(len(mnist_train))[:6000]
    test_indices = torch.randperm(len(mnist_test))[:1000]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(mnist_train, batch_size=batch_size, sampler=train_sampler, drop_last=True, pin_memory=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, sampler=test_sampler, drop_last=True, pin_memory=True)

    data, targets = next(iter(train_loader))
    print('data:', data.shape, 'targets:', targets.shape)

    #model = Lenet5().to(device)
    model = ResNet18().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    train_loss_hist = []
    test_acc_hist = []

    with torch.autograd.profiler.profile(use_cpu=True, record_shapes=True, profile_memory=False) as prof:
        for epoch in range(1):
            model.train()
            for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device)
                targets = targets.to(device)

                logits = model(data)   # logits:[batch_size, 10], targets:[batch_size]
                loss_val = loss(logits, targets)   # loss: tensor scaler, 长度为零的标量

                optimizer.zero_grad()
                loss_val.backward()
                optimizer.step()

            train_loss_hist.append(loss_val.item())
            print(f"Epoch {epoch}, Train Loss: {loss_val.item():.2f}")
            # 这里的loss为最后一个batch的loss，不能代表整个epoch训练过程，acc更能体现性能
            # Loss是一个标量，我们用item()将它转化为numpy，打印出来

            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_num = 0
                for data, targets in test_loader:
                    data = data.to(device)
                    targets = targets.to(device)

                    logits = model(data)  # [batch_size, 10]
                    pred = logits.argmax(dim=1)  #Return the index of the maximum value of input tensor
                    correct = torch.eq(pred, targets).float().sum().item()
                    total_correct += correct
                    total_num += data.shape[0]

                test_acc = total_correct / total_num
                test_acc_hist.append(test_acc)
                print(f"Epoch: {epoch}, Test Acc: {test_acc * 100: .2f}%\n")

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


    # # Plot test accuracy
    # fig = plt.figure(facecolor="w")
    # plt.plot(test_acc_hist)
    # plt.title("Accuracy Curves")
    # plt.legend(["Test Acc"])
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.show()

if __name__ == '__main__':
    main()

# with torch.autograd.profiler.profile(use_cpu=True, record_shapes=True, profile_memory=False) as prof:
#     with record_function("mnist_lenet5"):
#         mnist_lenet5()
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# with torch.cuda.profiler.profile():
#     mnist_lenet5()
#     with torch.autograd.profiler.emit_nvtx():
#         mnist_lenet5()