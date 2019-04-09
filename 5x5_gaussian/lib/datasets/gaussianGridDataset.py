import numpy as np
from torch.utils.data import Dataset

class gaussianGridDataset(Dataset):
    def __init__(self, n, n_data, sig):
        self.grid = np.linspace(-4, 4, n)
        self.data = None
        self.data_points = []
        for i in range(n):
            mean_x = self.grid[i]
            for j in range(n):
                mean_y = self.grid[j]
                self.data_points.append([mean_x, mean_y])
                if self.data is None:
                    self.data = np.random.multivariate_normal((mean_x, mean_y), cov=[[sig, 0.0], [0.0, sig]], size=n_data)
                else:
                    self.data = np.concatenate((self.data, np.random.multivariate_normal((mean_x, mean_y), cov=[[sig, 0.0], [0.0, sig]], size=n_data)), axis=0)

        self.out_dim = 2
        self.n_data = self.data.shape[0]
        self.name = "gaussian_grid"
        self.data_points = np.array(self.data_points)
    
    def get_num_modes(self):
        return 25

    def get_data_points(self):
        return self.data_points 

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.n_data
