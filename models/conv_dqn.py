import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device: {}".format(device))

def make_DQN(input_shape, output_shape):
    net = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, output_shape)
    )
    return net

class DuelingQNetwork(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        
        conv_output_size = 64 * 7 * 7
        
        self.value_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_shape) 
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) 
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        mean_advantage = advantage.mean(dim=1, keepdim=True)
        q_values = value + (advantage - mean_advantage)
        
        return q_values

def dueling_DQN(input_shape, output_shape):
    return DuelingQNetwork(input_shape, output_shape)
