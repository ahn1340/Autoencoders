import os
import torch

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# custom modules
from model import AutoEncoder
from dataset import IntelDataset
from utils import unnormalize, save_ims

# Datasets and loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize(size=(128, 128)),
])

train_dataset = IntelDataset(data_root='IntelDataset', mode='train', transform=transform)
test_dataset = IntelDataset(data_root='IntelDataset', mode='test', transform=transform)

train_loader = DataLoader(train_dataset,
                          batch_size=128,
                          shuffle=True,
                          )
test_loader = DataLoader(test_dataset,
                         batch_size=128,
                         shuffle=True,
                         )


if __name__=='__main__':
    #TODO: organize configs into a yamlfile
    # configs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 200
    save_freq = 5  # save im after how many epochs
    num_ims = 5  # how many images to save
    save_root = 'ckpt'
    os.makedirs(save_root, exist_ok=True)

    # model, loss function and optimizer
    model = AutoEncoder(hidden_dim=2048).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)

    # start training
    for epoch in range(1, epochs+1):
        train_loss_epoch = 0
        test_loss_epoch = 0
        # train
        for i, (image, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
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

        if epoch % save_freq == 0:
            # random sample
            idx = torch.randint(len(train_dataset), (num_ims,))
            samples = train_dataset.random_sampling(idx).to(device)
            output = model(samples)

            # save image
            samples = unnormalize(samples)
            output = unnormalize(output)
            save_ims(samples, output, epoch)

            # save model state dict
            save_path = os.path.join(save_root, f'autoencoder_epoch{epoch}')
            torch.save(model.state_dict(), save_path)





