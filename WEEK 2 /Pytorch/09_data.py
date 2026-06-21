# epoch = 1 forward and backwarrd pass of all the training samples
#batch_size = number of training samples in one forward and backward pass
#number of iterations = number of passes, each pass using [batch size] number of samples
# e.g 100 samples and batch size = 20 then 5 iterations for 1 epoch

import torch 
import torchvision
from torch.utils.data import Dataset , DataLoader
import numpy as np 
import math

class WineDataset(Dataset):

    def __init__(self):
        # data loading 
        xy = np.loadtxt ('location of text file in PC' , delimiter=',', dtype=np.float32, skiprows = 1)
        self.x = torch.from_numpy(xy[:, 1:]) #skip the first column and goes to end 
        self.y = torch.from_numpy(xy[:, [0]]) #all rows only column zero in different shape
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        #dataset[0] if someone asks for this give them the first of x and first of y
        return self.x[index], self.y[index] 
    
    def __len__(self):
        #len(dataset)
        return self.n_samples
    
dataset = WineDataset()
first_data = dataset[0]
features , labels = first_data

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle = True, num_workers=2)
datatite = iter(dataloader)
data = datatite.__next__()
features . labels = data

#training loop

num_epoch = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)

for epoch in range(num_epoch):
    for i, (inputs , labels) in enumerate(dataloader):
        #forward backward update 
        if(i+1)%5==0:
            print(f'epoch {epoch+1}/{num_epoch}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
            
