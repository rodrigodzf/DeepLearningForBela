# %%
# Code adapted from https://github.com/vsitzmann/siren

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
class SineLayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
                     
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., bias=True):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0, bias=bias))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,bias=bias))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0, bias=bias))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        return self.net(coords)


# %%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class WavDataset(Dataset):
    def __init__(self, path, timesteps):
        audio, sample_rate = torchaudio.load(path)

        audio = audio[..., :sample_rate]
        lenght = (len(audio.view(-1)) // timesteps) * timesteps 
        audio = audio[..., :lenght]

        x = torch.linspace(0, 1, lenght)
        self.batches_x = x.reshape(-1, timesteps, 1)
        self.batches_y = audio.reshape(-1, timesteps, 1)
    def __len__(self):
        return len(self.batches_x)

    def __getitem__(self, idx):
        return self.batches_x[idx], self.batches_y[idx]

dataset = WavDataset('../audio/voice.wav', 8192)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


batch_size = 2
timesteps = 8192

model = Siren(in_features=1, out_features=1, hidden_features=64, 
              hidden_layers=1, first_omega_0=3000, bias=False, outermost_linear=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 10000

# %%
for epoch in range(epochs):
    for model_input, target in dataloader:
        model_input = model_input.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        model_output = model(model_input)
        loss = F.mse_loss(model_output, target)
        loss.backward()
        optimizer.step()
print(f'final loss: {loss}')
# %%
eval_lenght = 44100
with torch.no_grad():
    test_input = torch.linspace(0, 1, eval_lenght, device=device).unsqueeze(-1)
    test_output = model(test_input)


# %% Plot the data in each partition in different colors:
plt.plot(test_input.cpu(), test_output.cpu(), label="Train")

# T, C = test_output.shape
test_output = test_output.permute(1, 0)
torchaudio.save('siren.wav', test_output.cpu(), 44100)
plt.legend()
plt.savefig('siren_pytorch.png')

# %% Set the model to inference mode
export_sequence_lenght = 128
model.eval()
model_name = f"siren{export_sequence_lenght}.onnx"
# Export the model
torch.onnx.export(model,                       # model being run
                  test_input[:export_sequence_lenght],      # model input (or a tuple for multiple inputs)
                  model_name,                  # where to save the model (can be a file or file-like object)
                  export_params=True,          # store the trained parameter weights inside the model file
                  opset_version=8,             # the ONNX version to export the model to
                  # do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],     # the model's input names
                  output_names = ['output'])   # the model's output names
# %%
