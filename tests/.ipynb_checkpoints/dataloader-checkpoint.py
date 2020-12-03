import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class GalaxyZooDatasetTrain(Dataset):
    
    def __init__(self, csv_file, root_dir, first_elem=0, last_elem=1):
        
        self.annotations = pd.read_csv(csv_file)
        self.index_shift = int(len(self.annotations)*first_elem)
        self.annotations = self.annotations[self.index_shift:int(len(self.annotations)*last_elem)]
        self.root_dir = root_dir
        
    def __len__(self):
        
        return len(self.annotations)*4
    
    def __getitem__(self, index):
        
        if index >= len(self.annotations)*4:
            print('dataset index ' + str(index + self.index_shift) + ' out of range ' + str(len(self.annotations)*4))
            raise IndexError('dataset index ' + str(index) + ' out of range')
        
        real_index, rotate_type = divmod(index, 4)
        
        img_path = os.path.join(self.root_dir, str(self.annotations["GalaxyID"][real_index + self.index_shift]) + ".jpg")
        image = io.imread(img_path)
        
        x_crop, y_crop = 96, 96
        x_point, y_point = (image.shape[0] - x_crop) // 2, (image.shape[1] - y_crop) // 2
        
        image = image[x_point:x_point + x_crop, y_point:y_point + y_crop]
        x_image = torch.as_tensor(image, dtype=torch.float32)
        
        x_image = torch.rot90(x_image, rotate_type)
        x_image = x_image.permute(2, 0, 1) 
        x_image = x_image.unsqueeze(0)
        
        y_lable = torch.tensor(self.annotations.iloc[real_index, 1:], dtype=torch.float32)
        
        return (x_image, y_lable)
    
class GalaxyZooDatasetTest(Dataset):
    
    def __init__(self, root_dir):
        
        self.root_dir = root_dir
        self.files_in_dir = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        
    def __len__(self):
        
        return len(self.files_in_dir)
    
    def __getitem__(self, index):
        
        if index >= len(self.files_in_dir):
            raise IndexError('dataset index ' + str(index) + ' out of range')
        
        img_path = os.path.join(self.root_dir, self.files_in_dir[index])
        image = io.imread(img_path)
        
        x_crop, y_crop = 96, 96
        x_point, y_point = (image.shape[0] - x_crop) // 2, (image.shape[1] - y_crop) // 2
        
        image = image[x_point:x_point + x_crop, y_point:y_point + y_crop]
        x_image = torch.as_tensor(image, dtype=torch.float32)
    
        x_image = x_image.permute(2, 0, 1) 
        x_image = x_image.unsqueeze(0)
        
        return self.files_in_dir[index][:-4], x_image

    
from albumentations import (
    RandomRotate90, Flip, Compose, Rotate, Crop
)
import numpy as np

def aug(prob=1):
    return Compose([
        RandomRotate90(p=1*prob),
        Flip(p=0.75*prob),
        Rotate(p=0.75*prob),
        Crop(x_min=149, x_max=245, y_min=149, y_max=245)
    ], p=1)

    
class GalaxyZooDatasetTrainV2(Dataset):
    
    def __init__(self, csv_file, root_dir, first_elem=0, last_elem=1, transform_prob=1):
        
        self.annotations = pd.read_csv(csv_file)
        self.index_shift = int(len(self.annotations)*first_elem)
        self.annotations = self.annotations[self.index_shift:int(len(self.annotations)*last_elem)]
        self.root_dir = root_dir
        self.transform_prob = transform_prob
        
    def __len__(self):
        
        return len(self.annotations)
    
    def __getitem__(self, index):
        
        if index >= len(self.annotations):
            print('dataset index ' + str(index + self.index_shift) + ' out of range ' + str(len(self.annotations)))
            raise IndexError('dataset index ' + str(index) + ' out of range')
        
        img_path = os.path.join(self.root_dir, str(self.annotations["GalaxyID"][index + self.index_shift]) + ".jpg")
        image = io.imread(img_path)
        augmentation = aug(prob=1)
        
        augmented = augmentation(**{"image":image})
        res =  augmented["image"]
        x_image = torch.as_tensor(res, dtype=torch.float32)
        
        x_image = x_image.permute(2, 0, 1) 
        x_image = x_image.unsqueeze(0)
        
        y_lable = torch.tensor(self.annotations.iloc[index, 1:], dtype=torch.float32)
        
        return (x_image, y_lable)
    
