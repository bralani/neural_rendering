import torch, json
import torch.nn as nn
import torch.optim as optim
import time
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
float_dtype = torch.float64


class RayDataset(Dataset):
    def __init__(self, file_path, batch_size = 512,range_size = 0.1):

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
          self.datas = self.datas[:100000]


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

        self.create_batch(batch_size, range_size)
        self.datas = []

    def create_batch(self, batch_size, range_size = 0.1):
        batch = [[[] for _ in range(int(1/range_size))] for _ in range(int(1/range_size))]

        for i in range(len(self.data)):
            dir = self.data[i][1]
            dir_x = int(torch.floor(dir[0] / range_size))
            dir_y = int(torch.floor(dir[1] / range_size))

            batch[dir_x][dir_y].append(self.data[i])

        self.batches = []
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                # divide in batches of batch_size
                for k in range(0, len(batch[i][j]), batch_size):
                    current_batch = batch[i][j][k:k+batch_size]
                    self.batches.append((current_batch, i, j))
        
    def get_embeddings(self):
        embeddings = []
        indices = [i for i in range(len(self.data))]
        
        indices = np.random.choice(indices, 10000, replace=False)
        labels = []
        for i in indices:
            embeddings.append(torch.cat([self.data[i][0], self.data[i][1]]))
            labels.append(self.data[i][2])

        embeddings = torch.stack(embeddings)
        labels = torch.stack(labels)

        return embeddings, labels

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class Encoder(nn.Module):
    def __init__(self, embeddings, labels_embedding):
        super(Encoder, self).__init__()
        self.embeddings = embeddings.to(float_dtype).to(device).requires_grad_(True)
        self.labels_embedding = labels_embedding.reshape(-1).to(device)
        self.labels_embedding[self.labels_embedding == 0] = -1

        self.num_embeddings = embeddings.shape[0]
        init_range = (0.1, 0.1)

        self.range_size = 0.1

        self.dim = 4
        
        self.cholesky_decomposition = nn.Parameter(
            torch.empty(self.num_embeddings, self.dim, self.dim, dtype=float_dtype, device=device).uniform_(*init_range),
            requires_grad=True
        )

        # Ensure that the cholesky decomposition is lower triangular
        self.cholesky_decomposition = nn.Parameter(torch.tril(self.cholesky_decomposition.data), requires_grad=True)
        self.eye_matrix = torch.eye(self.cholesky_decomposition.data.size(-1)).unsqueeze(0).expand_as(self.cholesky_decomposition.data)

        self.create_batch()


        
    def update_parameters(self):
        
        with torch.no_grad():
            L = torch.tril(self.cholesky_decomposition)

            epsilon = 1e-4

            # Ensure that the diagonal elements are positive
            L[:, torch.arange(L.size(-1)), torch.arange(L.size(-1))] = torch.abs(L[:, torch.arange(L.size(-1)), torch.arange(L.size(-1))]) + epsilon

            self.cholesky_decomposition.data = L

            self.create_batch()

    def create_batch(self):
        batch = [[[] for _ in range(int(1/self.range_size))] for _ in range(int(1/self.range_size))]
        with torch.no_grad():
            for idx in range(len(self.embeddings)):
                emb = self.embeddings[idx]
                dir_x = int(torch.floor(emb[2] / self.range_size))
                dir_y = int(torch.floor(emb[3] / self.range_size))

                batch[dir_x][dir_y].append(idx)


        self.batch_embeddings = batch

            
        
    def concatenate_embeddings_parameters(self, i, j):
    
        nearest_indices = []
        nearest_indices.extend(self.batch_embeddings[i][j])
        if i > 0:
            nearest_indices.extend(self.batch_embeddings[i-1][j])
        if i < len(self.batch_embeddings) - 1:
            nearest_indices.extend(self.batch_embeddings[i+1][j])
        if j > 0:
            nearest_indices.extend(self.batch_embeddings[i][j-1])
        if j < len(self.batch_embeddings[i]) - 1:
            nearest_indices.extend(self.batch_embeddings[i][j+1])

        embeddings = self.embeddings[nearest_indices]
        covariance_matrices = self.cholesky_decomposition[nearest_indices]
        labels_embedding = self.labels_embedding[nearest_indices]

        covariance_matrices = torch.matmul(covariance_matrices, covariance_matrices.permute(0, 2, 1))

        return embeddings, covariance_matrices, labels_embedding


    def forward(self, i, j):
        embeddings_parameters, covariance_matrices, labels_embedding = self.concatenate_embeddings_parameters(i, j)
        return embeddings_parameters, covariance_matrices, labels_embedding



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def pdf_2d(self,x, mu, Sigma, labels_embedding):
        """
        Calcola la funzione di densità di probabilità (pdf) di una distribuzione normale multivariata
        con matrice di covarianza usando PyTorch.

        Parametri:
        x (torch.Tensor): Tensore di variabili (dim = [N, 4]).
        mu (torch.Tensor): Tensore di media della distribuzione (dim = [M, 4]).
        Sigma (torch.Tensor): Tensore delle varianze della distribuzione (dim = [M, 4, 4]).

        Restituisce:
        torch.Tensor: PDF per ogni punto e gaussiana (dim = [N, M]).
        """

        diffs = x.unsqueeze(1) - mu.unsqueeze(0)
        diffs = diffs.unsqueeze(2)

        Sigma_inv = torch.inverse(Sigma)
        Sigma_inv = Sigma_inv.unsqueeze(0)

        pdf_values = -0.5 * (torch.sum(diffs @ Sigma_inv * diffs, dim=-1).squeeze(-1))
        pdf_values = torch.exp(pdf_values)
        pdf_values = (labels_embedding * pdf_values)
        
        return pdf_values


    def apply_gaussian_splatting(self, origins, directions, means, covariances, labels_embedding):
        
        # Crea un tensore delle coordinate degli origins
        pos = torch.cat([origins, directions], dim=1)  # (batch_size, 4)

        # Calcola le probabilità logaritmiche per tutte le gaussiane
        gaussian_values = self.pdf_2d(pos, means, covariances, labels_embedding)

        prob = torch.sum(gaussian_values, dim=1)
        prob = torch.sigmoid(prob)

        return prob


    def forward(self, origins, directions, means, covariances, labels_embedding):
        output = self.apply_gaussian_splatting(origins, directions, means, covariances, labels_embedding).reshape(-1, 1)

        return output



class Autoencoder(nn.Module):
    def __init__(self, embeddings, labels_embedding):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(embeddings, labels_embedding)
        self.decoder = Decoder()

    def forward(self, origins, directions, i, j):
        mean, covariance, labels_embedding = self.encoder(i, j)
        output = self.decoder(origins, directions, mean, covariance, labels_embedding)
        return output

    
def loss_function(predict, original_hits):
    loss = nn.BCELoss()(predict, original_hits)
    loss = loss
    return loss
    

# Funzione di test
def test_model(model, data_loader, device):
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch, i, j in  data_loader:

            origins = torch.stack([x[0] for x in batch])
            directions = torch.stack([x[1] for x in batch])
            hits = torch.stack([x[2] for x in batch])

            origins, directions, hits = origins.to(device), directions.to(device), hits.to(device)
            

            origins = origins.reshape(-1, 2)
            directions = directions.reshape(-1, 2)
            hits = hits.reshape(-1, 1)

            # Previsione
            reconstructed_hits = model(origins, directions, i, j)
            
            # Converti le previsioni e i valori reali in numpy arrays per la valutazione
            predictions = (reconstructed_hits > 0.5).float().cpu().numpy().flatten()  # Converti probabilità in classi
            true_labels = hits.cpu().numpy().flatten()
            
            all_predictions.extend(predictions)
            all_labels.extend(true_labels)

    return np.array(all_labels), np.array(all_predictions)



ray_dataset = RayDataset("/Users/matteobalice/Downloads/train_neural (2).json")
ray_data_loader = DataLoader(ray_dataset, batch_size=1, shuffle=True)

import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Hyperparameters
learning_rate = 0.001

# Initialize model, optimizer, and loss function
embeddings, labels_embedding = ray_dataset.get_embeddings()
model = Autoencoder(embeddings, labels_embedding)
optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
model.to(device)

# Assumiamo che il file di test sia simile a quello usato durante l'addestramento
test_file_paths = "/Users/matteobalice/Downloads/test_neural (2).txt"  # Aggiungi qui il percorso del tuo file di test
test_dataset = RayDataset(test_file_paths)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch, i, j in ray_data_loader:
        origins = torch.stack([x[0] for x in batch])
        directions = torch.stack([x[1] for x in batch])
        hits = torch.stack([x[2] for x in batch])

        origins, directions, hits = origins.to(device), directions.to(device), hits.to(device)

        origins = origins.reshape(-1, 2)
        directions = directions.reshape(-1, 2)
        hits = hits.reshape(-1, 1).to(float_dtype)

        optimizer.zero_grad()
        now = time.time()
        reconstructed_hits = model(origins, directions, i, j)
        print("Time for forward pass: ", time.time()- now)

        now = time.time()
        loss = loss_function(reconstructed_hits, hits)
        print("Time for loss computation: ", time.time()- now)

        now = time.time()
        loss.backward()
        print("Time for gradients computation: ", time.time()- now)


        now = time.time()
        optimizer.step()
        print("Time for optimizer step: ", time.time()- now)
        now = time.time()
        model.encoder.update_parameters()
        print("Time for update parameters: ", time.time()- now)
        total_loss += loss.item()
    

        

    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
    #if epoch % 20 == 0:
    model.eval()
    all_labels, all_predictions = test_model(model, test_data_loader, device)

    # Calcola e stampa le metriche di prestazione
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

