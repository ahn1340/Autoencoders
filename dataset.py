import os
import glob
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IntelDataset(Dataset):
    def __init__(self, data_root='IntelDataset', mode='train', transform=None):
        """
        Intel image classification dataset.
        Source: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
        :param data_root: root folder where data is stored. folder contains seg_pred, seg_train, seg_test
        :param mode: str, one of ['train', 'test']
        """
        self.data_root = data_root
        self.label_dict = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
        self.data = []
        folder_split = os.path.join(data_root, f'seg_{mode}', f'seg_{mode}')
        folders_class = glob.glob(folder_split + '/*')
        for folder in folders_class:
            label = self.label_dict[folder.split(os.sep)[-1]]
            self.data.extend([(file, label) for file in glob.glob(folder + '/*')])

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize(size=(128, 128)),
            ])  # standard transform TODO: add default augmentations
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        im = self.transform(im)

        return im, label

    def random_sampling(self, idx):
        """
        return a batch of len(idx) random samples
        :param idx: tuple of ints
        :return: torch tensor
        """
        assert hasattr(idx, '__iter__'), "idx should be an iterable of integers"
        samples = [self.__getitem__(i)[0] for i in idx]

        return torch.stack(samples, dim=0)
