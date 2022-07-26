import os
import torch
import matplotlib.pyplot as plt

def unnormalize(im, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    device = im.device
    mean = torch.tensor(mean, dtype=torch.float32).reshape(1, 3, 1, 1).to(device)
    std = torch.tensor(std, dtype=torch.float32).reshape(1, 3, 1, 1).to(device)

    return im * std + mean

def save_ims(ims, recons, epoch, folder_path='./reconstruction'):
    """
    :param ims: torch tensor, batch of images
    :param path: where to save the images
    :return: None
    """
    # detach images
    ims = ims.cpu().detach().numpy()
    recons = recons.cpu().detach().numpy()

    # reshape images s.t. channel dimension comes last
    ims = ims.transpose([0, 2, 3, 1])
    recons = recons.transpose([0, 2, 3, 1])

    # plot images
    nr, nc = 2, ims.shape[0]
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f'Epoch {epoch}')
    subfigs = fig.subfigures(2, 1)

    for r, subfig in enumerate(subfigs):
        if r == 0:
            subfigs[0].suptitle("Original")
            toplot = ims
        else:
            subfigs[1].suptitle("Reconstruction")
            toplot = recons

        axs = subfig.subplots(1, nc)
        for c, ax in enumerate(axs):
            ax.axis('off')
            ax.imshow(toplot[c])

    # create dir if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    fig.savefig(os.path.join(folder_path, f'Epoch{epoch}'))





