import torch 
import torchvision
from torch.utils.data import Dataset , DataLoader
import numpy as np 
import math

class WineDataset(Dataset):

    def __init__(self, transform=None):
        # data loading 
        xy = np.loadtxt ('location of text file in PC' , delimiter=',', dtype=np.float32, skiprows = 1)
        self.x = torch.from_numpy(xy[:, 1:]) #skip the first column and goes to end 
        self.y = torch.from_numpy(xy[:, [0]]) #all rows only column zero in different shape
        self.n_samples = xy.shape[0]
        self.transform = transform


    def __getitem__(self, index):
        #dataset[0] if someone asks for this give them the first of x and first of y
        sample = self.x[index], self.y[index] 
        
        if self.transform:
            sample = self.transprm(sample)

        return sample

    def __len__(self):
        #len(dataset)
        return self.n_samples
    
#tranform to tensor type array
class ToTensor:
    def __call__(self, sample):
        inputs , targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

dataset = WineDataset()