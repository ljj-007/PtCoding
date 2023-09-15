from ..utils import *

class TrainDataset(Dataset):
    def __init__(self, args, data):
        super(TrainDataset, self).__init__()
        self.args = args
        self.x = data.train_data[...]
        self.y = data.test_data[...]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class ValidDataset(Dataset):
    def __init__(self, args, data) -> None:
        super(ValidDataset, self).__init__()
        self.args = args
        self.x = data.test_data[...]
        self.y = data.test_data[...]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class TestDataset(Dataset):
    def __init__(self, args, data) -> None:
        super(TestDataset, self).__init__()
        self.args = args
        self.x = data.test_data[...]
        self.y = data.test_data[...]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]