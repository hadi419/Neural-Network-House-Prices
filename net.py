import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_hidden, activation, dropout):

        super().__init__()

        # Fixed attributes.
        self.activation = activation

        # Create layers.
        self.hidden_layers = nn.ModuleList()
        if num_hidden == 0:
            self.output_layer = nn.Linear(input_size, output_size)
        else:
            self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            for i in range(num_hidden - 1):
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
                if dropout > 0:
                    self.hidden_layers.append(nn.Dropout(dropout))
            self.output_layer = nn.Linear(hidden_size, output_size)
    
    
    def forward(self, x):

        for hidden_layer in self.hidden_layers:
            if self.activation == 'ReLU':
                x = F.relu(hidden_layer(x))
            elif self.activation == 'Sigmoid':
                x = F.sigmoid(hidden_layer(x))
            elif self.activation == 'Tanh':
                x = F.tanh(hidden_layer(x))
            else:
                raise Exception('Net.forward: invalid activation function.')
        
        y_hat = self.output_layer(x)
        return y_hat