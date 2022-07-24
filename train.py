import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, FashionMNIST
import matplotlib.pyplot as plt

from model import AutoEncoder

transform = transforms.Compose([
    transforms.ToTensor()
])

cifar10_train = CIFAR10(root='CIFAR10/train',
                        train=True,
                        transform=transform,
                        target_transform=None,
                        download=True,
                        )
cifar10_test = CIFAR10(root='CIFAR10/test',
                       train=False,
                       transform=transform,
                       target_transform=None,
                       download=True,
                       )
fashion_train = FashionMNIST(root='MNIST/train',
                             train=True,
                             transform=transform,
                             target_transform=None,
                             download=True,
                             )
fashion_test = FashionMNIST(root='MNIST/test',
                            train=False,
                            transform=transform,
                            target_transform=None,
                            download=True,
                            )

train_loader = DataLoader(cifar10_train,
                          batch_size=64,
                          shuffle=True,
                          )
test_loader = DataLoader(cifar10_test,
                         batch_size=64,
                         shuffle=True,
                         )



if __name__=='__main__':
    device = "cuda:0"
    model = AutoEncoder(hidden_dim=128).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    epochs = 50
    for epoch in range(1, epochs+1):
        train_loss_epoch = 0
        test_loss_epoch = 0
        # train
        for i, (image, target) in enumerate(train_loader):
            image = image.to(device)
            output = model(image)

            loss = criterion(image, output)
            train_loss_epoch += loss.detach().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # validate
        with torch.no_grad():
            for i, (image, target) in enumerate(test_loader):
                image = image.to(device)
                output = model(image)

                loss = criterion(image, output)
                test_loss_epoch += loss.detach().item()
        print(f"[Epoch {epoch}] train loss: {train_loss_epoch}")
        print(f"[Epoch {epoch}] test loss: {test_loss_epoch}")


        if (epoch) % 5 == 0:
            for i, (image, target) in enumerate(test_loader):
                image = image.to(device)
                output = model(image)
                sample = output[0].cpu().detach()
                plt.figure()
                plt.imshow(image[0].cpu().permute(1, 2, 0))
                plt.show()
                plt.figure()
                plt.imshow(sample.permute(1, 2, 0))
                plt.show()
                break
