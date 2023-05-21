import torch
import os
import cv2
import numpy as np

mapping = {'разметка_шипун' : 2, 'klikun' : 1, 'разметка_малый':0}
inv_mapping = { 2 : 'разметка_шипун', 1 : 'klikun', 0:'разметка_малый'}

class SwanTestDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_folder, transform=None, mapping = mapping):
        self.path_to_folder = path_to_folder
        self.transform = transform
        self.mapping = mapping
        self.images = os.listdir(self.path_to_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.path_to_folder, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'image_name' : img_name}
        return sample
    
class MultiHeadTestSwanDataset(torch.utils.data.Dataset):
    def __init__(self, images, crops, heads, transform=None, mapping = mapping):
        self.images = images
        self.crops = crops
        self.heads = heads
        self.transform = transform
        self.mapping = mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        heads = self.heads[idx]
        crops =  self.crops[idx]
        if len(heads) > 0:
            for i in range(len(heads)):
                crops[i] = np.array(crops[i]).astype(np.uint8)
                heads[i] = np.array(heads[i]).astype(np.uint8)
        else:
            crops = [np.zeros((224,224,3)).astype(np.uint8)]
            heads = [np.zeros((224,224,3)).astype(np.uint8)]
        
        if self.transform:
            image = self.transform(image)
            for i in range(len(heads)):
                crops[i] = self.transform(crops[i])
                heads[i] = self.transform(heads[i])
        sample = {'image': image,'heads':heads,'crops':crops}
        return sample