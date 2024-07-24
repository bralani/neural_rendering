import torch, json
import torch.nn as nn
import torch.optim as optim
import time

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.distributions import MultivariateNormal
device = torch.device('cpu')


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

          np.random.shuffle(self.datas)
          self.datas = self.datas[:10000]


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
          self.datas = []

    def get_embeddings(self):
        embeddings = []
        # cycle all data where label = 1
        for (origin, dir, label) in self.data:
            if label == 1:
                embeddings.append(torch.cat([origin, dir]))

        embeddings = torch.stack(embeddings)

        return embeddings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        origins, directions, hits = self.data[idx]

        return (origins, directions, hits)


class Encoder(nn.Module):
    def __init__(self, embeddings):
        super(Encoder, self).__init__()
        self.embeddings = embeddings

        num_embeddings = embeddings.shape[0]
        dim = 2
        init_range = (0.1, 0.1)


        # parameters (sigma_x, sigma_y)
        self.embeddings_parameters = nn.Parameter(
            torch.empty(num_embeddings, dim).uniform_(init_range[0], init_range[1]),
            requires_grad=True
        )

        self.proxy_variable = nn.Parameter(
            torch.empty(num_embeddings, 1).uniform_(0.4, 0.8),
            requires_grad=True)


            
        
    def concatenate_embeddings_parameters(self):

        proxy_variables_sigmoid = torch.sigmoid(self.proxy_variable)
        proxy_variables_binary = (proxy_variables_sigmoid > 0.5)
        indices = torch.where(proxy_variables_binary == 1)[0]


        # take only the embeddings_parameters with proxy_variables_binary = 1
        embeddings_parameters = self.embeddings_parameters[indices]
        embeddings = self.embeddings[indices]

        return torch.cat([embeddings, embeddings_parameters], dim=1), proxy_variables_sigmoid


    def forward(self, x):
        embeddings_parameters, proxy = self.concatenate_embeddings_parameters()
        return embeddings_parameters, proxy



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()


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
  
        
    def pdf_max(self, mu, Sigma):
        """
        Calcola il valore massimo della pdf di una distribuzione normale multivariata
        con media mu e matrice di covarianza Sigma per più gaussiane.

        Parametri:
        mu (torch.Tensor): Tensor di media (dim = [N, d]).
        Sigma (torch.Tensor): Tensor di matrice di covarianza (dim = [N, d, d]).

        Restituisce:
        torch.Tensor: Valori massimi della pdf per ciascuna gaussiana (dim = [N]).
        """
        N, d = mu.shape
        
        # Calcola il determinante della matrice di covarianza
        det_Sigma = torch.prod(Sigma, dim=1)
        
        # Calcola il fattore di normalizzazione per ogni gaussiana
        normalization_factor = (2 * torch.pi) ** (d / 2) * torch.sqrt(det_Sigma)
        
        # Calcola il valore massimo della pdf per ogni gaussiana
        pdf_max_values = 1 / normalization_factor
        
        return pdf_max_values

    def pdf_2d(self,x, mu, Sigma_diag):
        """
        Calcola la funzione di densità di probabilità (pdf) di una distribuzione normale multivariata
        con matrice di covarianza diagonale in 2D usando PyTorch.

        Parametri:
        x (torch.Tensor): Tensore di variabili (dim = [N, 2]).
        mu (torch.Tensor): Tensore di media della distribuzione (dim = [N, M, 2]).
        Sigma_diag (torch.Tensor): Tensore delle varianze della distribuzione (dim = [N, M, 2]).

        Restituisce:
        torch.Tensor: PDF per ogni punto e gaussiana (dim = [N, M]).
        """
        N, D = x.shape
        _, M, _ = mu.shape

        # Espandi x per fare broadcasting
        x = x.unsqueeze(1).expand(-1, M, -1)  # [N, M, 2]

        # Calcola il termine della densità
        det_Sigma = torch.prod(Sigma_diag, dim=-1)  # Determinante della matrice di covarianza [N, M]
        diff = x - mu  # [N, M, 2]
        exponent = -0.5 * torch.sum((diff ** 2) / Sigma_diag, dim=-1)  # [N, M]

        # Calcola la pdf
        normalization_factor = (2 * torch.pi) ** (D / 2) * torch.sqrt(det_Sigma)  # [N, M]
        pdf = torch.exp(exponent) / normalization_factor  # [N, M]

        return pdf


    def apply_gaussian_splatting(self, origins, directions, latents):
        
        # Crea un tensore delle coordinate degli origins
        pos = torch.cat([origins, directions], dim=1)  # (batch_size, 4)
        
        mean_x = latents[:, 0]
        mean_y = latents[:, 1]
        mean_dir_x = latents[:, 2]
        mean_dir_y = latents[:, 3]
        sigma_x = torch.relu(latents[:, 4]) + 0.01
        sigma_y = torch.relu(latents[:, 5]) + 0.01
        sigma_dir_x = torch.full((latents.shape[0],), 0.001)
        sigma_dir_y = torch.full((latents.shape[0],), 0.001)

        # Espandi i means e le sigmas per il broadcasting such that they have the same shape as pos
        means = torch.stack([mean_x, mean_y, mean_dir_x, mean_dir_y], dim=-1).reshape(-1, 4)  # (num_gaussians, 4)
        sigmas = torch.stack([sigma_x, sigma_y, sigma_dir_x, sigma_dir_y], dim=-1).reshape(-1, 4)  # (num_gaussians, 4)

        pdf_max_values = self.pdf_max(means, sigmas)

        means = means.unsqueeze(0).expand(pos.shape[0], -1, -1)  # (batch_size, num_gaussians, 4)
        sigmas = sigmas.unsqueeze(0).expand(pos.shape[0], -1, -1)  # (batch_size, num_gaussians, 4)

        # Calcola le probabilità logaritmiche per tutte le gaussiane
        gaussian_values = self.pdf_2d(pos, means, sigmas) / pdf_max_values

        prob = torch.sum(gaussian_values, dim=1)
        prob = torch.clamp(prob, 0, 1)

        return prob


    def forward(self, origins, directions, latents):
        output = self.apply_gaussian_splatting(origins, directions, latents).reshape(-1, 1)

        return output



class Autoencoder(nn.Module):
    def __init__(self, embeddings):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(embeddings)
        self.decoder = Decoder()

    def forward(self, origins, directions):
        latents, proxy = self.encoder(origins)
        output = self.decoder(origins, directions, latents)
        return output, proxy

    
def loss_function(predict, original_hits, proxy):

    num_positive_samples = torch.sum(original_hits)
    num_negative_samples = original_hits.shape[0] - num_positive_samples

    weight_positive = num_negative_samples / (num_positive_samples + num_negative_samples)

    # Calcola la perdita BCE pesata
    loss = nn.BCELoss(weight=torch.tensor([weight_positive]))(predict, original_hits)


    # Calcola la BCE Loss come usuale
    bce_loss = nn.BCELoss(weight=torch.tensor([weight_positive]))(predict, original_hits)
    
    # Penalizzazione delle previsioni di valori vicini a target_negative per le classi negative
    negative_mask = (original_hits == 0).float()
    
    # Penalizzazione per valori delle classi negative vicini a 0.4999
    target_neg_loss = negative_mask * torch.abs(predict - 0.4999)
    
    # Calcola la perdita totale
    loss = bce_loss + target_neg_loss.mean()

    #loss = nn.BCELoss()(predict, original_hits)

    #proxy_binary = (proxy > 0.5).float()

    
    threshold = 0.8
    proxy_mean = torch.mean(proxy)

    proxy_var_max = threshold*(1-threshold)
    proxy_variance = torch.var(proxy)

    loss2 = torch.abs(proxy_mean - threshold) + torch.abs(proxy_variance - proxy_var_max)


    # Combinare le perdite
    loss = loss + loss2
    return loss
    

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
            reconstructed_hits, _ = model(origins, directions)
            
            # Converti le previsioni e i valori reali in numpy arrays per la valutazione
            predictions = (reconstructed_hits > 0.5).float().cpu().numpy().flatten()  # Converti probabilità in classi
            true_labels = hits.cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_labels.extend(true_labels)

    return np.array(all_labels), np.array(all_predictions)



ray_dataset = RayDataset("/Users/matteobalice/Downloads/train_neural (2).json")
ray_data_loader = DataLoader(ray_dataset, batch_size=512)

import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Hyperparameters
learning_rate = 0.001

# Initialize model, optimizer, and loss function
embeddings = ray_dataset.get_embeddings()
model = Autoencoder(embeddings)
optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
model.to(device)

# Assumiamo che il file di test sia simile a quello usato durante l'addestramento
test_file_paths = "/Users/matteobalice/Downloads/test_neural (2).txt"  # Aggiungi qui il percorso del tuo file di test
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
        reconstructed_hits, proxy = model(origins, directions)

        embeddings = model.encoder.embeddings_parameters
        loss = loss_function(reconstructed_hits, hits, proxy)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    

        

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
    #if epoch % 20 == 0:
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

