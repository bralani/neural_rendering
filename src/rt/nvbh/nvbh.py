import json
import os
import time
from matplotlib import pyplot as plt
import psutil
import torch
import torch.nn as nn
import numpy as np
from gridencoder import GridEncoder
from torch.utils.data import Dataset
import numpy as np
import tracemalloc
import pygame, time
import numpy as np
from pygame.locals import *
import torch
from camera import show_camera
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.nn as nn
import numpy as np
from local_cache import generate_cache


# Avvia il monitoraggio dell'allocazione della memoria
tracemalloc.start()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_points = 200

def convert_spherical_to_cartesian(theta, phi):
    sin_theta = torch.sin(theta)

    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    
    return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim=-1)

t_values = torch.linspace(0, 1, n_points).reshape(1, n_points, 1).to(device)
def sample_points_along_ray(point1, point2):
    
    diff = point2 - point1
    length = torch.norm(diff, dim=-1, keepdim=True).clamp(1e-6, 1e6)
    points = point1[:, None, :] + diff[:, None, :]  * t_values
    directions = diff / length

    return points, directions


class RenderDatasetSph(Dataset):
    def __init__(self,data_dir="",transform=None) -> None:
        super().__init__()

        self.transform = transform
        self.data = []
        self.test_datas = []

        # Apri il file in modalità lettura
        with open(data_dir, 'r') as file:
            # Leggi tutte le righe del file
            lines = file.readlines()

            # Apri il file in modalità lettura
            with open(data_dir, 'r') as file:
                # Leggi tutte le righe del file
                lines = file.readlines()

                np.random.shuffle(lines)
                test_data = lines[:100000]
                train_data = lines[100000:]
                #train_data = train_data[:5000000]
                self.data = train_data
                self.test_datas = test_data

        del lines

    def data_cache(self):
        lines = self.data[:2000000]
        
        lines = "\n".join(lines)
        data = np.fromstring(lines, sep=' ').reshape(-1, 9)

        para = torch.tensor(data[:, :4], device=device, dtype=torch.float32)
        labels = torch.tensor(data[:, 4], device=device, dtype=torch.float32).view(-1, 1)
        rgb = torch.tensor(data[:, 5:8], device=device, dtype=torch.float32) / 255
        dist = torch.tensor(data[:, -1], device=device, dtype=torch.float32).view(-1, 1)

        return para, labels, rgb, dist

    def test_data(self):
        return self.test_datas
    
    def rendering(self):
        num_samples = 1920 * 1080
        np.random.shuffle(self.data)
        return self.data[:num_samples]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
    
class NeuralNetwork(nn.Module):
    def __init__(self, cache):
        super().__init__()

        self.cache = cache
        self.level_dim = 1
        self.num_levels = 6
        base_resolution = 16
        self.encoder_label = GridEncoder(input_dim=3, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=base_resolution)

        self.model_label = nn.Sequential(
            nn.Linear(self.num_levels, 1),
            nn.Sigmoid()
        )

        self.level_dim2 = 4
        self.num_levels2 = 6
        base_resolution2 = 16
        self.encoder_rgb = GridEncoder(input_dim=3, num_levels=self.num_levels2, level_dim=self.level_dim2, base_resolution=base_resolution2)

        self.model_rgb = nn.Sequential(
            nn.Linear(3 + self.num_levels2 * (self.level_dim2), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.num_points = 1
        self.arange_points = torch.arange(self.num_points, device=device)

    def setup(self, x):
        
        if x.size(0) != self.num_points:
            self.num_points = x.size(0)
            self.arange_points = torch.arange(self.num_points, device=device)
    
    
    def predict(self, x):

        self.setup(x)

        with torch.no_grad():
            point1 = convert_spherical_to_cartesian(x[:, 0], x[:, 1])
            point2 = convert_spherical_to_cartesian(x[:, 2], x[:, 3])
            points_encoded, directions = sample_points_along_ray(point1, point2)
            
            output = self.model_label(self.encoder_label(points_encoded))
            output = (output > 0.5).float()

            indices_first = torch.argmax(output, dim=1).view(-1)
            points_encoded = points_encoded[self.arange_points, indices_first]
            points_encoded = self.encoder_rgb(points_encoded)
            points_encoded = torch.cat([directions, points_encoded], dim=-1)

            output, _ = torch.max(output, dim=1)
            output_rgb = self.model_rgb(points_encoded)

        del points_encoded, directions

        return output, output_rgb
    
    def forward(self, x):

        self.setup(x)

        with torch.no_grad():
            point1 = convert_spherical_to_cartesian(x[:, 0], x[:, 1])
            point2 = convert_spherical_to_cartesian(x[:, 2], x[:, 3])
            points_encoded, directions = sample_points_along_ray(point1, point2)

        output = self.model_label(self.encoder_label(points_encoded))
        output_hits, _ = torch.max(output, dim=1)

        with torch.no_grad():
            indices_first = torch.argmax((output > 0.5).float(), dim=1).view(-1)
            points_encoded = points_encoded[self.arange_points, indices_first]

        points_encoded = self.encoder_rgb(points_encoded)
        points_encoded = torch.cat([directions, points_encoded], dim=-1)
        output_rgb = self.model_rgb(points_encoded)

        del points_encoded, directions

        return output_hits, output, indices_first, output_rgb

def loss_fn(output_label, labels, dist, all_outputs, output_rgb, rgb):

    mask = (labels > 0.5).squeeze()

    dist_indices = torch.clamp((dist[mask] * n_points).long().view(-1) - 1, 0, n_points - 2)

    rows = torch.arange(n_points).expand(all_outputs[mask].size(0), n_points).to(device)
    mask2 = rows <= dist_indices.unsqueeze(1)

    prev_hit = all_outputs[mask][mask2]
    loss0 = nn.BCELoss()(prev_hit, torch.zeros_like(prev_hit))

    dist_indices = torch.clamp((dist[mask] * n_points).long().view(-1), 0, n_points - 1)
    mask3 = rows == dist_indices.unsqueeze(1)
    hit = all_outputs[mask][mask3]
    loss1 = nn.BCELoss()(hit, torch.ones_like(hit))

    non_hit = all_outputs[~mask]
    loss2 = nn.BCELoss()(non_hit, torch.zeros_like(non_hit))

    # rgb loss
    loss3 = nn.MSELoss()(output_rgb[mask], rgb[mask])

    return loss0 + loss1 + loss2 + loss3


def get_memory_usage():
    # Ottieni l'uso della memoria del processo corrente
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Utilizza la memoria fisica residente (RSS)

def train():
    # Device configuration

    dataset = RenderDatasetSph(data_dir="C:/Users/miche/Downloads/test_neural_bunny.txt")
    dataset_loader = DataLoader(dataset,batch_size=2**13,shuffle=True)

    para_cache, labels_cache, _, dist_cache = dataset.data_cache()
    cache = generate_cache(n_points, para_cache, dist_cache,labels_cache)

    # Controlla lo stato della memoria iniziale
    print(f"Memory usage: {get_memory_usage()}")
    model = NeuralNetwork(cache).to(device)
    print(f"Memory usage: {get_memory_usage()}")



    test_set = []
    test_set_labels = []
    test_set_rgb = []
    test_set_dist = []
    lines = dataset.test_data()
    # Per ogni riga del file
    for line in lines:
        
        
        line = line.strip()
        tokens = line.split()
        tokens = [float(token) for token in tokens]

        points = torch.Tensor(tokens[:4])
        label = torch.Tensor([tokens[4]])
        rgb = torch.Tensor(tokens[5:8]) / 255
        dist = torch.Tensor([tokens[-1]])

        test_set.append(points)
        test_set_rgb.append(rgb)
        test_set_labels.append(label)
        test_set_dist.append(dist)

    test_set = torch.stack(test_set).to(device)
    test_set_labels = torch.tensor(test_set_labels).to(device)
    test_set_rgb = torch.stack(test_set_rgb).to(device)
    test_set_dist = torch.stack(test_set_dist).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    # Train the model
    total_step = len(dataset_loader)

    add_time_total = 0
    for epoch in range(4):
        epoch_train_loss = 0

        add_time = 0
        
        for i, lines in enumerate(dataset_loader):
           
            lines = "\n".join(lines)
            data = np.fromstring(lines, sep=' ').reshape(-1, 9)

            para = torch.tensor(data[:, :4], device=device, dtype=torch.float32)
            labels = torch.tensor(data[:, 4], device=device, dtype=torch.float32).view(-1, 1)
            rgb = torch.tensor(data[:, 5:8], device=device, dtype=torch.float32) / 255
            dist = torch.tensor(data[:, -1], device=device, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()

            now = time.time()
            output_label, all_outputs, _, output_rgb = model(para)
            add_time += time.time() - now

            loss1 = loss_fn(output_label, labels, dist, all_outputs, output_rgb, rgb)
            loss1.backward()
            optimizer.step()

            epoch_train_loss += loss1.item()


            if i % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch+1, 100, i+1, total_step, loss1.item()))
                
        add_time_total += add_time
        print(f"Add time: {add_time}")

        model.eval()

        all_preds = []
        all_preds_dist = []
        all_preds_rgb = []
        with torch.no_grad():
            for i in range(0, len(test_set), 4096):
                output, _, dist, output_rgb = model(test_set[i:i+4096])
                all_preds.append(output)
                all_preds_rgb.append(output_rgb)
                all_preds_dist.append((dist / n_points).view(-1, 1))

        all_preds = torch.cat(all_preds, dim=0)
        all_preds_dist = torch.cat(all_preds_dist, dim=0)
        all_preds_rgb = torch.cat(all_preds_rgb, dim=0)

        #remove the preds_rgb that have not been hit
        mask = (all_preds > 0.5).squeeze()
        
        all_preds_dist = all_preds_dist[mask]
        all_dist_true = test_set_dist[mask]
        all_preds_rgb = all_preds_rgb[mask]
        all_rgb_true = test_set_rgb[mask]

        
        all_preds[all_preds > 0.5] = 1
        all_preds[all_preds <= 0.5] = 0
        all_preds = all_preds.cpu()
        all_labels = test_set_labels.cpu()


        f1 = f1_score(all_labels, all_preds)
        dist_loss = nn.L1Loss()(all_preds_dist, all_dist_true).item()
        rgb_loss = nn.L1Loss()(all_preds_rgb, all_rgb_true).item()


        print("MODEL 1")
        print(f"F1: {f1}")
        print(f"Dist L1: {dist_loss}")
        print(f"RGB L1: {rgb_loss}")
        
        diff = torch.abs(all_preds_dist - all_dist_true)
        percentili = torch.quantile(diff, torch.tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99], device=device))
        max = torch.max(diff)
        print(f"Percentili test: {percentili}")
        
        print(f"Max diff test: {max}")

    show_camera(model)
    
train()
