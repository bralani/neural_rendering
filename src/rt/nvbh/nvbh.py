import torch
import torch.nn as nn
import numpy as np
from gridencoder import GridEncoder
from torch.utils.data import Dataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_points = 200

def convert_spherical_to_cartesian(theta, phi):
    
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    
    return torch.stack((x, y, z), dim=-1)

t_values = torch.linspace(0, 1, n_points).reshape(1, n_points, 1).to(device)
def sample_points_along_ray(int1, int2):

    point1 = convert_spherical_to_cartesian(int1[:, 0], int1[:, 1])
    point2 = convert_spherical_to_cartesian(int2[:, 0], int2[:, 1])
    
    N = point1.size(0)
    diff = (point2 - point1)
    point1 = point1.reshape(N, 1, 3)
    diff = diff.reshape(N, 1, 3)

    # (N, n_points, 3)
    points = point1 + diff * t_values
    
    return points.reshape(N, n_points, 3)


class RenderDatasetSph(Dataset):
    def __init__(self,data_dir="",transform=None) -> None:
        super().__init__()

        self.transform = transform
        
        self.data = []

        # Apri il file in modalità lettura
        with open(data_dir, 'r') as file:
            # Leggi tutte le righe del file
            lines = file.readlines()

            np.random.shuffle(lines)
            #lines = lines[:1000000]
            self.data = lines

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # multi-resolution hash grid
        self.encoder = GridEncoder(input_dim=3, num_levels=4, level_dim=8, base_resolution=32)

        
        # very simple model -> just a linear layer to map the output of the encoder to a single probability value
        self.model = nn.Sequential(
            nn.Linear(4 * 8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        num_shape = x.size(0)

        # sampling n_points along the ray
        with torch.no_grad():
            points = sample_points_along_ray(x[:, :2], x[:, 2:4])

        # encode the points
        t = self.encoder(points).reshape(num_shape * n_points, -1)

        # get the probability for each point
        h = self.model(t).reshape(num_shape, n_points)

        # get the maximum probability along the ray points
        h = torch.max(h, dim=1)[0].view(-1, 1)

        return h

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def loss_fn(output1, output2, labels):

    loss1 = nn.BCELoss()(output1, labels)
    loss2 = nn.BCELoss()(output2, labels)
    return loss1 + loss2


def train():
    # Device configuration
    model = NeuralNetwork().to(device)

    # Load the dataset
    dataset = RenderDatasetSph(data_dir="c:/Users/miche/Downloads/train_neural_26M.txt")
    dataset_loader = DataLoader(dataset,batch_size=4096,shuffle=True)

    # Load the dataset datas/test.txt with open
    file_path = 'c:/Users/miche/Downloads/test_neural.txt'


    test_set = []
    test_set_labels = []
    # Apri il file in modalità lettura
    with open(file_path, 'r') as file:
        # Leggi tutte le righe del file
        lines = file.readlines()

        # Per ogni riga del file
        for line in lines:
            # Rimuovi gli spazi bianchi iniziali e finali
            line = line.strip()
            # Dividi la riga in token
            tokens = line.split()
            # Converti i token in numeri
            tokens = [float(token) for token in tokens]
            # Aggiungi la riga alla lista


            '''
            tokens[0] /= torch.pi
            tokens[1] = (tokens[1] + torch.pi) / (2*torch.pi)
            tokens[2] /= torch.pi
            tokens[3] = (tokens[3] + torch.pi) / (2*torch.pi)
            tokens[4] /= torch.pi
            tokens[5] /= torch.pi'''


            points = torch.Tensor(tokens[:4])

            test_set.append(points)
            test_set_labels.append(tokens[-1])

    test_set = torch.stack(test_set).to(device)
    test_set_labels = torch.tensor(test_set_labels).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

    # Train the model
    total_step = len(dataset_loader)
    for epoch in range(5):
        epoch_train_loss = 0

        for i, lines in enumerate(dataset_loader):
            
            lines = "\n".join(lines)
            data = np.fromstring(lines, sep=' ').reshape(-1, 5)

            para = torch.tensor(data[:, :4], device=device, dtype=torch.float32)
            labels = torch.tensor(data[:, -1], device=device, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()
            output1 = model(para)

            para2 = torch.cat((para[:, 2:4], para[:, :2]), dim=1)
            output2 = model(para2)

            loss1 = loss_fn(output1, output2, labels)
            loss1.backward()
            optimizer.step()

            epoch_train_loss += loss1.item()

            if (i + 1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, 100, i+1, total_step, loss1.item()))
                
        model.eval()

        all_preds = []
        with torch.no_grad():
            # batches of 4096
            for i in range(0, len(test_set), 4096):
                output = model(test_set[i:i+4096])
                all_preds.append(output)

        # Convert list of tensors to a single tensor
        all_preds = torch.cat(all_preds, dim=0)

        all_preds[all_preds > 0.5] = 1
        all_preds[all_preds <= 0.5] = 0
        all_preds = all_preds.cpu()
        all_labels = test_set_labels.cpu()


        f1 = f1_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        print(f"F1: {f1}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

      
    model.eval()
        
    # Load the dataset datas/test.txt with open
    file_path = 'c:/Users/miche/Downloads/test_neural_image.txt'

    image = []
    # Apri il file in modalità lettura
    with open(file_path, 'r') as file:
        # Leggi tutte le righe del file
        lines = file.readlines()

        i = 0
        # Per ogni riga del file
        for line in lines:
            # Rimuovi gli spazi bianchi iniziali e finali
            line = line.strip()
            # Dividi la riga in token
            tokens = line.split()
            # Converti i token in numeri
            tokens = [float(token) for token in tokens]

            # Aggiungi la riga alla lista
            
            if(tokens[0] == 100):
                image.append(float(0.0))
            else:
                
                points = torch.tensor(tokens[:4]).reshape(1, 4).to(device)
                output = model(points).reshape(1).cpu().detach().numpy()[0]

                if output > 0.5:
                    image.append(float(1.0))
                else:
                    image.append(float(0.0))
            i += 1

    image[0] = float(0.0)
    image[1] = float(1.0)

    matrix = np.array(image).reshape((512, 512))
    matrix_flipped_180 = np.rot90(matrix, 2)

    plt.imshow(matrix_flipped_180, cmap='gray')
    plt.axis('off')
    plt.show()

train()
