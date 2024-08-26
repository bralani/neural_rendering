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
    
    norm = torch.norm(diff, dim=-1).unsqueeze(-1)
    norm = torch.clamp(norm, 1e-6, 1e6)
    directions = diff / norm

    return points, directions


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
        
        self.level_dim = 1
        self.num_levels = 6
        base_resolution = 8
        self.encoder_label = GridEncoder(input_dim=3, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=base_resolution)

        self.model_label = nn.Sequential(
            nn.Linear(self.num_levels, 1),
            nn.Sigmoid()
        )

        self.level_dim2 = 4
        self.num_levels2 = 5
        base_resolution2 = 4
        self.encoder_rgb = GridEncoder(input_dim=3, num_levels=self.num_levels2, level_dim=self.level_dim2, base_resolution=base_resolution2)

        self.model_rgb = nn.Sequential(
            nn.Linear(3 + self.num_levels2 * (self.level_dim2), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def predict(self, x):
        return self.grids(x, self.model, self.encoder)

    
    def forward(self, x, dist_true = None):

        num_points = x.size(0)
        with torch.no_grad():
            points_encoded, directions = sample_points_along_ray(x[:, :2], x[:, 2:4])

        output = self.encoder_label(points_encoded).reshape(num_points, int(n_points), -1)
        output = self.model_label(output)
        output_hits, _ = torch.max(output, dim=1)

        
        # get the first voxel hitted
        points_encoded = self.encoder_rgb(points_encoded)

        with torch.no_grad():
            # get the first voxel hitted
            all_labels = (output > 0.5).float()
            indices_first = torch.argmax(all_labels, dim=1).view(-1)
            indices_first = torch.clamp(indices_first, 2, n_points - 3)

            num_points2 = torch.arange(num_points)
        points_encoded = points_encoded[num_points2, indices_first]

        
        # add x to points_encoded
        points_encoded = torch.cat([directions, points_encoded], dim=-1)

        output_rgb = self.model_rgb(points_encoded)

        return output_hits, output, indices_first, output_rgb
    
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import numpy as np


decay_rate = 1e-3
indices_decay = torch.arange(int(n_points), dtype=torch.float32, device=device)
decay = -decay_rate * indices_decay
max_decay = torch.max(decay)
stable_decay = decay - max_decay
exponential_decay = torch.exp(stable_decay).to(device)

def find_index_with_exponential_decay(tensor):
    # Creare un vettore di maschere usando la funzione sigmoide
    threshold = 0.5
    masks = torch.sigmoid(50*(tensor - threshold))
    
    # Calcolare le probabilità esponenziali normalizzate
    probs = exponential_decay * masks
    sum = torch.sum(probs, dim=1).unsqueeze(1)
    probs = probs / sum
    
    # Calcolare l'indice ponderato
    weighted_index = torch.sum(indices_decay * probs, dim=1)
    
    return weighted_index

def loss_fn(output_label, labels, dist, all_outputs, output_rgb, rgb):

    mask = (labels > 0.5).squeeze()

    # Applichiamo la soglia morbida
    indices_first = find_index_with_exponential_decay(all_outputs.view(-1, int(n_points))).view(-1, 1) / (int(n_points))
    
    # L1 Loss tra gli indici soft e dist
    loss1 = nn.L1Loss()(indices_first[mask], dist[mask])
    loss0 = nn.BCELoss()(output_label, labels)
    loss2 = nn.MSELoss()(output_rgb, rgb)

    return loss0 + loss1 + loss2


def get_memory_usage():
    # Ottieni l'uso della memoria del processo corrente
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Utilizza la memoria fisica residente (RSS)

def train():
    # Device configuration

    dataset = RenderDatasetSph(data_dir="C:/Users/miche/Downloads/test_neural_rgb (1).txt")
    dataset_loader = DataLoader(dataset,batch_size=2**13,shuffle=True)

    # Controlla lo stato della memoria iniziale
    print(f"Memory usage: {get_memory_usage()}")
    model = NeuralNetwork().to(device)
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
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


    # Train the model
    total_step = len(dataset_loader)

    add_time_total = 0
    for epoch in range(5):
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
                output, _, dist, output_rgb = model(test_set[i:i+4096], test_set_dist[i:i+4096])
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
    
    show_camera(model)
    
train()
