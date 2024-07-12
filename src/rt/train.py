import torch, json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# SET HERE YOUR VARIABILES #
size_neurons = 256          # Number of neurons in the hidden layers 
batch_size = 512            # 512 is suitable for the GPU, if you have a CPU you should use a smaller batch size
path_train_set = "/content/drive/MyDrive/train.json" # Path to the training set
path_test_set = "/content/drive/MyDrive/test.txt" # Path to the test set
# ------------------------- #

# Device configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class RenderDatasetSph(Dataset):
    def __init__(self,data_dir="datas/sph_1.json",transform=None) -> None:
        super().__init__()

        self.transform = transform

        with open(data_dir) as f:
            self.datas = json.load(f)

        self.data = []
        for data in self.datas:

            para = torch.Tensor(data["point_sph"] + data["dir_sph"])
            label = torch.Tensor([data["label"]])

            self.data.append((para, label))

    def __getitem__(self, index):
        para, label = self.data[index]

        if self.transform is not None:
            para = self.transform(para)
            label = self.transform(label)

        return para, label

    def __len__(self) -> int:
        return len(self.data)


class NeuralNetwork(nn.Module):
    def __init__(self, size=64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(4, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, 1),
            # binary classification with sigmoid activation
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.model(x)


def loss_fn(output1, output2, target):
    loss1 = nn.BCELoss()(output1, target)
    loss2 = nn.BCELoss()(output2, target)
    return loss1 + loss2

def train():
    model = NeuralNetwork(size_neurons).to(device)
    dataset = RenderDatasetSph(data_dir=path_train_set)
    dataset_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    max_acc = 0

    test_set = []
    test_set_labels = []
    with open(path_test_set, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()
            tokens = line.split()
            tokens = [float(token) for token in tokens]

            test_set.append(tokens[:-1])
            test_set_labels.append(tokens[-1])

    test_set = torch.tensor(test_set).to(device)
    test_set_labels = torch.tensor(test_set_labels).to(device)

    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

    # Train the model
    total_step = len(dataset_loader)
    for epoch in range(200):
        for i, (para, labels) in enumerate(dataset_loader):

            # Move tensors to the configured device
            para = para.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            output1 = model(para)

            # exchange the inputs 0 and 1 with 2 and 3
            para2 = para.clone()
            para2[:,0], para2[:,1], para2[:,2], para2[:,3] = para[:,2], para[:,3], para[:,0], para[:,1]
            output2 = model(para2)

            loss = loss_fn(output1, output2, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, 100, i+1, total_step, loss.item()))

        all_preds = []
        with torch.no_grad():
            all_preds = model(test_set)

        all_preds[all_preds > 0.5] = 1
        all_preds[all_preds <= 0.5] = 0
        all_preds = all_preds.cpu()
        all_labels = test_set_labels.cpu()

        f1 = f1_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        
        if acc > max_acc:
            max_acc = acc
            example_input = torch.rand(2,4).to(device)
            traced_script_module = torch.jit.trace(model, example_input)
            traced_script_module.save("model_sph.pt")

        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)

        print(f"F1: {f1}")
        print(f"Accuracy: {acc}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")


    model.eval()

train()