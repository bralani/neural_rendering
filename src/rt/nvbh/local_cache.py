import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

grid_size = 60
size_hashmap = 300000

def hash_function(a, b, c, d, size):
    
    prime = 31
    
    # Combinazione di a, b, c, d usando operazioni bitwise
    hash_value = (a ^ (b << 5) ^ (c << 10) ^ (d << 15)) % size
    
    # Ulteriore mixing con il primo elemento
    hash_value = (hash_value * prime) % size
    
    return hash_value


# Funzione per generare i dati di esempio
def generate_data(X, continuous_y, M):
    
    cell_attive = {}
    cell_attive2 = {}
    hash_map = torch.zeros(size_hashmap, M, device=device)
    hash_map2 = torch.zeros(size_hashmap, M, device=device)
    hash_map_final = torch.zeros(size_hashmap, M, device=device)

    X2 = X.clone()
    X2[:, 0] = ((X2[:, 0] / torch.pi) * (grid_size - 1)).long()
    X2[:, 1] = (((X2[:, 1] + torch.pi) / (2*torch.pi)) * (grid_size - 1)).long()
    X2[:, 2] = ((X2[:, 2] / torch.pi) * (grid_size - 1)).long()
    X2[:, 3] = (((X2[:, 3] + torch.pi) / (2*torch.pi)) * (grid_size - 1)).long()

    y_discrete = (continuous_y * M).long()

    for idx in range(len(X2)):
        i = X2[idx, 0].long()
        j = X2[idx, 1].long()
        k = X2[idx, 2].long()
        l = X2[idx, 3].long()
        hash_code = hash_function(i, j, k, l, size_hashmap)
        hash_map[hash_code, y_discrete[idx].long()] = 1
        cell_attive[(int(i), int(j), int(k), int(l))] = 1

    for (i, j, k, l) in cell_attive.keys():
        cell_attive2[i, j, k, l] = 1 
        cell_attive2[(i+1) % grid_size, j, k, l] = 1 
        cell_attive2[(i-1) % grid_size, j, k, l] = 1 
        cell_attive2[i, (j+1) % grid_size, k, l] = 1
        cell_attive2[i, (j-1) % grid_size, k, l] = 1
        cell_attive2[i, j, (k+1) % grid_size, l] = 1
        cell_attive2[i, j, (k-1) % grid_size, l] = 1
        cell_attive2[i, j, k, (l+1) % grid_size] = 1
        cell_attive2[i, j, k, (l-1) % grid_size] = 1

    for (i, j, k, l) in cell_attive2.keys():
        tensors = torch.stack([
            hash_map2[hash_function(i, j, k, l, size_hashmap)], 
            hash_map[hash_function(i, j, k, l, size_hashmap)], 
            hash_map[hash_function((i+1) % grid_size, j, k, l, size_hashmap)], 
            hash_map[hash_function((i-1) % grid_size, j, k, l, size_hashmap)], 
            hash_map[hash_function(i, (j+1) % grid_size, k, l, size_hashmap)],
            hash_map[hash_function(i, (j-1) % grid_size, k, l, size_hashmap)],
            hash_map[hash_function(i, j, (k+1) % grid_size, l, size_hashmap)],
            hash_map[hash_function(i, j, (k-1) % grid_size, l, size_hashmap)],
            hash_map[hash_function(i, j, k, (l+1) % grid_size, size_hashmap)],
            hash_map[hash_function(i, j, k, (l-1) % grid_size, size_hashmap)],
        ])
        tensors = torch.any(tensors, dim=0).int()
        hash_map2[hash_function(i, j, k, l, size_hashmap)] = tensors

    sum = torch.sum(hash_map2, dim=1).view(-1)
    sum = sum[sum > 0]
    percentili = torch.quantile(sum, torch.tensor([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999], device=device))
    print("percentili: ")
    print(percentili)
    max = int(percentili[-2])

    hash_map_final = torch.zeros(size_hashmap, int(max), device=device)

    for i in range(size_hashmap):
        indices = torch.nonzero(hash_map2[i] == 1).flatten()

        mancanti = max - len(indices)
        if mancanti >= 0:
            tensor = torch.cat([torch.zeros(mancanti, device=device), indices])
        elif mancanti < 0:
            indices_rand = torch.randperm(indices.size(0))[:max]
            tensor = indices[indices_rand]
            
        hash_map_final[i] = tensor

    return hash_map_final



def generate_cache(M, para, dist, labels):
        
    labels = labels.view(-1)
    para = para[labels == 1]
    dist = dist[labels == 1]

    cache = generate_data(para, dist, M)

    return cache


