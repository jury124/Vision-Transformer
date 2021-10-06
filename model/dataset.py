import os
from PIL import Image
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, phase='train'):
        
        self.phase = phase
        self.data_path = data_path
        self.data = pd.read_csv(
                os.path.join(data_path,f'{self.phase}.csv')
                )
        self.num_data = len(self.data)
        self.transform = transform

    def __getitem__(self, index):
#        print(self.data_path,f'/{self.phase}/{self.data.file[index]}')
        image = Image.open(f'{self.data_path}/{self.phase}/{self.data.file[index]}').convert('RGB')
        # image = cv2.imread(os.path.join(self.data_path,f'/{self.phase}/{self.data.file[index]}'))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image)
        else:
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        if self.phase == 'test':
            label = self.data.label[index]
            return image, label, self.data.file[index]
        else:
            label = self.data.label[index]
            return image, label

    def __len__(self):
        return self.num_data
