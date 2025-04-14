from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class FashionMNISTLoader:
    def __init__(self, batch_size=64, val_split=0.0, data_dir='./data'):
        self.batch_size = batch_size
        self.val_split = val_split
        self.data_dir = data_dir

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def load_datasets(self):
        train_dataset = datasets.FashionMNIST(root=self.data_dir, train=True, download=True, transform=self.transform)
        test_dataset = datasets.FashionMNIST(root=self.data_dir, train=False, download=True, transform=self.transform)

        if self.val_split > 0:
            val_size = int(self.val_split * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        else:
            self.val_loader = None

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        self.load_datasets()
        return self.train_loader, self.val_loader, self.test_loader
