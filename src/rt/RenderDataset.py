import torch
from torch.utils.data import Dataset
import json
import random

class RenderDatasetSph(Dataset):
    def __init__(self,data_dir="datas/sph_1.json",transform=None) -> None:
        super().__init__()

        self.transform = transform

        # Carica i dati dal file JSON
        with open(data_dir) as f:
            self.datas = json.load(f)


        # Separa i dati in due liste per classe
        self.class0_data = []
        self.class1_data = []

        for data in self.datas:
            
            data["point_sph"][0] /= torch.pi
            data["point_sph"][1] /= 2*torch.pi
            data["dir_sph"][0] /= torch.pi
            data["dir_sph"][1] /= 2*torch.pi
            
            para = torch.Tensor(data["point_sph"] + data["dir_sph"])
            label = torch.Tensor([data["label"]])

            if label == 0:
                self.class0_data.append((para, label))
            elif label == 1:
                self.class1_data.append((para, label))

        '''
        # Sottocampiona la classe negativa in modo che entrambe le classi abbiano lo stesso numero di esempi
        num_samples = min(len(self.class0_data), len(self.class1_data))
        self.class0_data = random.sample(self.class0_data, num_samples)
        self.class1_data = random.sample(self.class1_data, num_samples)'''

        # Unisci i dati bilanciati
        self.balanced_data = self.class0_data + self.class1_data



    def __getitem__(self, index):
        para, label = self.balanced_data[index]

        if self.transform is not None:
            para = self.transform(para)
            label = self.transform(label)

        return para, label

    def __len__(self) -> int:
        return len(self.balanced_data)
