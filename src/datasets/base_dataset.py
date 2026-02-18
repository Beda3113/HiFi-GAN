from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base class for all datasets"""
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError