import os
import glob
import cv2

from torch.utils.data import Dataset

class IntelDataset(Dataset):
    def __init__(self, data_root='IntelDataset', mode='train'):
        """
        Intel image classification dataset.
        Source: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
        :param data_root: root folder where data is stored. folder contains seg_pred, seg_train, seg_test
        :param mode: str, one of ['train', 'test']
        """
        self.data_root = data_root
        self.label_dict = {'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}
        self.data = []
        folder_split = os.path.join(data_root, f'seg_{mode}', f'seg_{mode}')
        folders_class = glob.glob(folder_split + '/*')
        for folder in folders_class:
            label = self.label_dict[folder.split(os.sep)[-1]]
            self.data.extend([(os.path.join(folder, file), label) for file in glob.glob(folder + '/*')])
            #TODO: apply transform; add features such as train/val split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        im = cv2.imread(path)
        #TODO: [WIP] implement data loading
