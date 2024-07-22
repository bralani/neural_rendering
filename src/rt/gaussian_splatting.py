import torch, json
import torch.nn as nn
import torch.optim as optim

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.distributions import MultivariateNormal


class RayDataset(Dataset):
    def __init__(self, file_path):

        if not file_path.endswith(".json"):
          self.data = []
          with open(file_path, 'r') as file:
              lines = file.readlines()

              for line in lines:
                  line = line.strip()
                  tokens = line.split()
                  tokens = [float(token) for token in tokens]

                  tokens[0] /= torch.pi
                  tokens[1] = (tokens[1] + torch.pi) / (2*torch.pi)
                  tokens[2] /= torch.pi
                  tokens[3] = (tokens[3] + torch.pi) / (2*torch.pi)
                  tokens[4] /= torch.pi
                  tokens[5] /= torch.pi

                  origin = torch.Tensor(tokens[:2])
                  dir = torch.Tensor(tokens[4:6])
                  label = torch.Tensor([tokens[-1]])

                  self.data.append((origin, dir, label))
        else:
          with open(file_path) as f:
              self.datas = json.load(f)

          #np.random.shuffle(self.datas)
          #self.datas = self.datas[:10000]


          self.data = []
          for data in self.datas:


              data["point1_sph"][0] /= torch.pi
              data["point1_sph"][1] = (data["point1_sph"][1] + torch.pi) / (2*torch.pi)
              data["point2_sph"][0] /= torch.pi
              data["point2_sph"][1] = (data["point2_sph"][1] + torch.pi) / (2*torch.pi)
              data["dir_sph"][0] /= torch.pi
              data["dir_sph"][1] /= torch.pi

              origin = torch.Tensor(data["point1_sph"])
              dir = torch.Tensor(data["dir_sph"])
              label = torch.Tensor([data["label"]])

              self.data.append((origin, dir, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        origins, directions, hits = self.data[idx]

        return (origins, directions, hits)


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        latent = self.fc5(x)
        return latent



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.model = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # binary classification with sigmoid activation
            nn.Sigmoid()
        )

    def log_prob_2d(self, x, mu, Sigma_diag):
      """
      Calcola il logaritmo della funzione di densità di probabilità (pdf) di una distribuzione normale multivariata
      con matrice di covarianza diagonale in 2D usando PyTorch.

      Parametri:
      x (torch.Tensor): Tensore di variabili (dim = [N, 2]).
      mu (torch.Tensor): Tensore di media della distribuzione (dim = [N, M, 2]).
      Sigma_diag (torch.Tensor): Tensore delle varianze della distribuzione (dim = [N, M, 2]).

      Restituisce:
      torch.Tensor: Logaritmo della pdf per ogni punto e gaussiana (dim = [N,M]).
      """
      N, D = x.shape
      _, M, _ = mu.shape

      # Espandi x per fare broadcasting
      x = x.unsqueeze(1).expand(-1, M, -1)  # [N, M, 2]

      # Calcola il termine del logaritmo della densità
      log_det = torch.sum(torch.log(Sigma_diag), dim=-1)  # [N, M]
      diff = x - mu  # [N, M, 2]
      exponent = -0.5 * torch.sum((diff ** 2) / Sigma_diag, dim=-1)  # [N, M]

      # Calcola il logaritmo della funzione di densità di probabilità
      log_pdf = -0.5 * D * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * log_det + exponent  # [N, M]

      return log_pdf
  
    def apply_gaussian_splatting(self, origins, latents):

      # Estrai le coordinate x e y degli origins
      x = origins[:, 0].unsqueeze(1)  # (batch_size, 1)
      y = origins[:, 1].unsqueeze(1)  # (batch_size, 1)
      
      # Crea un tensore delle coordinate degli origins
      pos = torch.cat([x, y], dim=1)  # (batch_size, 2)
      
      mean_x = latents[:, 0::5]
      mean_y = latents[:, 1::5]
      sigma_x = torch.exp(latents[:, 2::5])
      sigma_y = torch.exp(latents[:, 3::5])

      # use tanh to ensure weights are between -1 and 1  
      weights = torch.tanh(latents[:, 4::5])


      # Espandi i means e le sigmas per il broadcasting
      means = torch.stack([mean_x, mean_y], dim=-1)  # (batch_size, num_gaussians, 2)
      sigmas = torch.stack([sigma_x, sigma_y], dim=-1)  # (batch_size, num_gaussians, 2)


      # Calcola le probabilità logaritmiche per tutte le gaussiane
      gaussian_values = self.log_prob_2d(pos, means, sigmas)
      gaussian_values = weights * gaussian_values

      # Somma i valori logaritmici per ogni gaussiano
      total_loss = torch.sum(gaussian_values, dim=1)

      return total_loss

    def forward_output(self, origins, directions):
        x = torch.cat([origins, directions], dim=1)
        x = self.model(x)
        return x

    def forward(self, origins, directions, latents):
        output1 = self.forward_output(origins, directions)
        output2 = self.apply_gaussian_splatting(origins, latents).reshape(-1, 1)

        weight_output1 = (4/3)*output1**3 - (4/3)*output1 + 1
        weight_output2 = (output2*2 - 1)

        output = weight_output1 + (1-weight_output1) * weight_output2

        return output



class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    def forward(self, origins, directions):
        latent = self.encoder(directions)
        output = self.decoder(origins, directions, latent)
        return output
    
def loss_function(predict, original_hits):
    predictions = nn.Sigmoid()(predict)
    #l2_norm_predict = torch.norm(predict, p=2, dim=1)
    #lam = 0.01
    return nn.BCELoss()(predictions, original_hits) #+ lam * l2_norm_predict.mean()
    

# Funzione di test
def test_model(model, data_loader, device):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for (origins, directions, hits) in data_loader:
            origins, directions, hits = origins.to(device), directions.to(device), hits.to(device)
            

            origins = origins.reshape(-1, 2)
            directions = directions.reshape(-1, 2)
            hits = hits.reshape(-1, 1)

            # Previsione
            reconstructed_hits = model(origins, directions)
            
            # Converti le previsioni e i valori reali in numpy arrays per la valutazione
            predictions = (reconstructed_hits > 0.5).float().cpu().numpy().flatten()  # Converti probabilità in classi
            true_labels = hits.cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_labels.extend(true_labels)

    return np.array(all_labels), np.array(all_predictions)



ray_dataset = RayDataset("/content/drive/MyDrive/train_neural.json")
ray_data_loader = DataLoader(ray_dataset, batch_size=512, shuffle=True)

import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Hyperparameters
learning_rate = 1e-3

# parameters for each gaussian
num_gaussians = 20 # Gaussian per direction
latent_dim = num_gaussians * 5  # 2 mean + 2 sigmas + 1 weight

# Initialize model, optimizer, and loss function
model = Autoencoder(latent_dim)
optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Assumiamo che il file di test sia simile a quello usato durante l'addestramento
test_file_paths = "/content/drive/MyDrive/test_neural.txt"  # Aggiungi qui il percorso del tuo file di test
test_dataset = RayDataset(test_file_paths)
test_data_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
  model.train()
  total_loss = 0
  for (origins, directions, hits) in ray_data_loader:
      origins, directions, hits = origins.to(device), directions.to(device), hits.to(device)

      origins = origins.reshape(-1, 2)
      directions = directions.reshape(-1, 2)
      hits = hits.reshape(-1, 1)

      optimizer.zero_grad()
      reconstructed_hits = model(origins, directions)
      loss = loss_function(reconstructed_hits, hits)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()

  print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
  if epoch % 20 == 0:
    model.eval()
    all_labels, all_predictions = test_model(model, ray_data_loader, device)

    # Calcola e stampa le metriche di prestazione
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

