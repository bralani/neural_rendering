import torch
from simpleNet import NeuralNetwork
from RenderDataset import RenderDatasetSph
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import torch.nn as nn
import numpy as np

def loss_fn(output1, output2, target):
    loss1 = nn.BCELoss()(output1, target)
    loss2 = nn.BCELoss()(output2, target)
    return loss1 + loss2

def train():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(4, 256).to(device)
    dataset = RenderDatasetSph(data_dir="datas/train.json")
    dataset_loader = DataLoader(dataset,batch_size=512,shuffle=True)

    # Load the dataset datas/test.txt with open
    file_path = 'datas/test.txt'

    max_acc = 0

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
            
            tokens[0] /= torch.pi 
            tokens[1] /= 2*torch.pi 
            tokens[2] /= torch.pi 
            tokens[3] /= 2*torch.pi 
            
            test_set.append(tokens[:-1])
            test_set_labels.append(tokens[-1])

    test_set = torch.tensor(test_set).to(device)
    test_set_labels = torch.tensor(test_set_labels).to(device)


    criterion = nn.BCELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

    # Train the model
    total_step = len(dataset_loader)
    for epoch in range(200):
        for i, (para, labels) in enumerate(dataset_loader):
            # Move tensors to the configured device
            para = para.to(device)
            labels = labels.to(device)

            # Forward pass
            output1 = model(para)

            # exchange the inputs 0 and 1 with 2 and 3
            para2 = para.clone()
            para2[:,0], para2[:,1], para2[:,2], para2[:,3] = para[:,2], para[:,3], para[:,0], para[:,1]
            output2 = model(para2)

            loss = loss_fn(output1, output2, labels)
            

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, 100, i+1, total_step, loss.item()))

        all_preds = []
        with torch.no_grad():
            for para in test_set:
                outputs = model(para)
                all_preds.append(outputs.item())

        all_preds = np.array(all_preds)
        all_preds[all_preds > 0.5] = 1
        all_preds[all_preds <= 0.5] = 0
        all_labels = test_set_labels.cpu().numpy()



        acc = accuracy_score(all_labels, all_preds)
        if acc > max_acc:
            max_acc = acc
            example_input = torch.rand(2,4).to(device)
            traced_script_module = torch.jit.trace(model, example_input)
            traced_script_module.save("models/model_sph1.pt")

        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")


    model.eval()

train()