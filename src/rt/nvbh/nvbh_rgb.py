import json
import os
import time
import psutil
import torch
import torch.nn as nn
import numpy as np
from gridencoder import GridEncoder
from torch.utils.data import Dataset
import numpy as np
import tracemalloc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from camera import show_camera


# Avvia il monitoraggio dell'allocazione della memoria
tracemalloc.start()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_points = 200
# Gestione della camera e del rendering
range_size = 0.2


def convert_spherical_to_cartesian(theta, phi):
    sin_theta = torch.sin(theta)

    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    
    return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim=-1)

t_values = torch.linspace(0, 1, n_points).reshape(1, n_points, 1).to(device)
def sample_points_along_ray(int1, int2):

    point1 = convert_spherical_to_cartesian(int1[:, 0], int1[:, 1])
    point2 = convert_spherical_to_cartesian(int2[:, 0], int2[:, 1])
    
    diff = point2 - point1
    points = point1.unsqueeze(1) + diff.unsqueeze(1) * t_values
    
    return points



class RenderDatasetSph(Dataset):
    def __init__(self,data_dir="",transform=None) -> None:
        super().__init__()

        self.transform = transform
        self.data = []
        self.test_datas = []

        with open(data_dir, 'r') as file:
            lines = file.readlines()

            np.random.shuffle(lines)
            test_data = lines[:100000]
            train_data = lines[100000:]
            train_data = train_data[:1000000]
            self.data = train_data
            self.test_datas = test_data

        del lines

    def test_data(self):
        return self.test_datas

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.level_dim = 4
        self.num_levels = 3
        base_resolution = 8
        self.encoder = GridEncoder(input_dim=3, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=base_resolution)

        self.model_label = nn.Sequential(
            nn.Linear(self.num_levels, 1),
            nn.Sigmoid()
        )
        self.model_rgb = nn.Sequential(
            nn.Linear(self.num_levels * (self.level_dim-1), 3),
            nn.Sigmoid()
        )

    def predict(self, x):
        return self.grids(x, self.model, self.encoder)


    def forward(self, x, labels = None):

        with torch.no_grad():
            output = sample_points_along_ray(x[:, :2], x[:, 2:4])
        output = self.encoder(output)
        input_labels = output[:, :, ::self.level_dim]
        all_prob = self.model_label(input_labels)
        output_hits, _ = torch.max(all_prob, dim=1)

        mask = torch.ones(output.size(2), dtype=bool)
        mask[::self.level_dim] = False
        input_rgb = output[:, :, mask]
        
        # get the first voxel hitted
        all_labels = (all_prob > 0.5).float()
        indices_first = torch.argmax(all_labels, dim=1).unsqueeze(1)
        
        # get the features of the first voxel hitted
        input_rgb = torch.gather(input_rgb, 1, indices_first.expand(-1, -1, input_rgb.size(2))).view(-1, input_rgb.size(2))

        output_rgb = self.model_rgb(input_rgb)


        return output_hits, output_rgb

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import numpy as np

def loss_fn(output_label, output_rgb, labels, rgb):

    loss1 = nn.BCELoss()(output_label, labels)
    loss2 = nn.L1Loss()(output_rgb, rgb)
    return loss1 + loss2


def get_memory_usage():
    # Ottieni l'uso della memoria del processo corrente
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Utilizza la memoria fisica residente (RSS)

def train():
    # Device configuration

    dataset = RenderDatasetSph(data_dir="C:/Users/miche/Downloads/test_neural_rgb.txt")
    dataset_loader = DataLoader(dataset,batch_size=2**13,shuffle=True)

    # Controlla lo stato della memoria iniziale
    print(f"Memory usage: {get_memory_usage()}")
    model = NeuralNetwork().to(device)
    print(f"Memory usage: {get_memory_usage()}")


    test_set = []
    test_set_labels = []
    test_set_rgb = []
    lines = dataset.test_data()
    # Per ogni riga del file
    for line in lines:
        
        
        line = line.strip()
        tokens = line.split()
        tokens = [float(token) for token in tokens]

        points = torch.Tensor(tokens[:4])
        label = torch.Tensor([tokens[4]])
        rgb = torch.Tensor(tokens[5:8])

        test_set.append(points)
        test_set_rgb.append(rgb)
        test_set_labels.append(label)

    test_set = torch.stack(test_set).to(device)
    test_set_labels = torch.tensor(test_set_labels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


    # Train the model
    total_step = len(dataset_loader)

    add_time_total = 0
    for epoch in range(5):
        epoch_train_loss = 0

        add_time = 0

        for i, lines in enumerate(dataset_loader):
           
            lines = "\n".join(lines)
            data = np.fromstring(lines, sep=' ').reshape(-1, 8)

            para = torch.tensor(data[:, :4], device=device, dtype=torch.float32)
            labels = torch.tensor(data[:, 4], device=device, dtype=torch.float32).view(-1, 1)
            rgb = torch.tensor(data[:, 5:], device=device, dtype=torch.float32) / 255

            optimizer.zero_grad()

            now = time.time()
            output_label, output_rgb = model(para, labels)
            add_time += time.time() - now

            loss1 = loss_fn(output_label, output_rgb, labels, rgb)
            loss1.backward()
            optimizer.step()

            epoch_train_loss += loss1.item()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch+1, 100, i+1, total_step, loss1.item()))
                
        add_time_total += add_time
        print(f"Add time: {add_time}")

        model.eval()

        all_preds = []
        with torch.no_grad():
            for i in range(0, len(test_set), 4096):
                output, _ = model(test_set[i:i+4096])
                all_preds.append(output)

        all_preds = torch.cat(all_preds, dim=0)

        all_preds[all_preds > 0.5] = 1
        all_preds[all_preds <= 0.5] = 0
        all_preds = all_preds.cpu()
        all_labels = test_set_labels.cpu()


        f1 = f1_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        print("MODEL 1")
        print(f"F1: {f1}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
    
    show_camera(model)
    
train()
