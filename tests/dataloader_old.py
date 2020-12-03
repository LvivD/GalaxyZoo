import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io

class GalaxyZooDataset(Dataset):
    def __init__(self, csv_file, root_dir):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
    def __len__(self):
        return len(self.annotations)*4
    
    def __getitem__(self, index):
        
        real_index, rotate_type = divmod(index, 4)
        
        img_path = os.path.join(self.root_dir, str(self.annotations["GalaxyID"][real_index]) + ".jpg")
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

