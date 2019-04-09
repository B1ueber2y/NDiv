from datasets.gaussianGridDataset import gaussianGridDataset
from datasets.gaussianRingDataset import gaussianRingDataset

def getDataset(dataset_config):
    dataset_name = dataset_config['name']
    if dataset_name == "gaussian_grid":
        return gaussianGridDataset(dataset_config['n'], dataset_config['n_data'], dataset_config['sig'])
    elif dataset_name == "gaussian_ring":
        return gaussianRingDataset(dataset_config['n'], dataset_config['n_data'], dataset_config['sig'])
    else:
        raise ValueError("no such dataset called " + dataset_name)

