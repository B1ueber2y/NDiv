import numpy as np
from torch.utils.data import Dataset
import math

class gaussianRingDataset(Dataset):
    def __init__(self, n, n_data, sig):
        self.grid = list(np.linspace(0, 1, n))
        self.radius = 2
        self.data_points = [[self.radius * math.cos(j * np.pi * 2 / n), self.radius * math.sin(j * np.pi * 2 / n)] for j in range(n)] 
        self.data = None

        for point in self.data_points:
            mean_x = point[0]
            mean_y = point[1]
            if self.data is None:
                self.data = np.random.multivariate_normal((mean_x, mean_y), cov=[[sig, 0.0], [0.0, sig]], size=n_data)
            else:
                self.data = np.concatenate((self.data, np.random.multivariate_normal((mean_x, mean_y), cov=[[sig, 0.0], [0.0, sig]], size=n_data)), axis=0)

        self.out_dim = 2
        self.n_data = self.data.shape[0]
        self.name = "gaussian_ring"
        self.data_points = np.array(self.data_points)

    def get_num_modes(self):
        return 8
    
    def get_data_points(self):
        return self.data_points 

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data

