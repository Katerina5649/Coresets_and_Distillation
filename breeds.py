from torch.utils.data import Dataset
import os
from PIL import Image

import torchvision.transforms as T
import torch

class Breeds(Dataset):
    def __init__(
        self,
        root="/home/dev/data_main/CORESETS/breeds",
        indices=None,
        train=True,
        resize=224,
        scores_data = scores_data
    ):
        self.scores_data = scores_data
        self.root = root
        self.resize = resize
        self.train = train
        self.transform = (
            T.Compose(
                [
                    T.Resize((self.resize, self.resize)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    T.RandomAffine(
                        degrees=5, scale=(0.8, 1.2), translate=(0.2, 0.2)
                    ),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            if self.train
            else T.Compose(
                [
                    T.Resize((self.resize, self.resize)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        )
        

        if self.train:
            _list_f = os.path.join(self.root, "train_labels.txt")
        else:
            _list_f = os.path.join(self.root, "val_labels.txt")
        self.pathes = []
        with open(_list_f, "r") as lines:
            for line in lines:
                self.pathes += [line]
                
        self.targets = []
        for i in self.pathes:
            self.targets += [int(i.split()[-1])]
            
        self.indices = (
            indices if indices is not None else np.arange(len(self.pathes))
        )

    def __getitem__(self, index):
        index = self.indices[index]
        path = self.pathes[index].split()
        image, label = path[0], int(path[-1])
        image = Image.open(image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), self.scores_data[index]

    def __len__(self):
        return len(self.pathes)