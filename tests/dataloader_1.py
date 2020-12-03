import os 
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as transforms

class GalaxyZooDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(self.annotations["GalaxyID"][index]) + ".jpg")
        image = io.imread(img_path)
        y_lable = torch.tensor(self.annotations.iloc[index, 1:])
        
        if self.transform:
            image = self.transform(image)
            
        return (image, y_lable)
    
if __name__ == "__main__":
    data_csv = "../data/training_solutions_rev1/training_solutions_rev1.csv"
    root_dir = "../data/images_training_rev1"
    my_dataset = GalaxyZooDataset(csv_file=data_csv, root_dir=root_dir, transform=transforms.ToTensor())
    print(my_dataset[0])