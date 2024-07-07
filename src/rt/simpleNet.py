from torch import nn
class NeuralNetwork(nn.Module):
    def __init__(self, input, size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input, size),
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