# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.onnx
device = 'cpu'

class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(in_features, hidden_features),
                                    nn.ReLU(),
                                    nn.Linear(hidden_features,
                                              hidden_features),
                                    nn.ReLU(),
                                    nn.Linear(hidden_features, out_features))

    def forward(self, x):
        return self.linear(x)


batch_size = 128
timesteps = 128
uniform = torch.distributions.Uniform(torch.tensor([-1.0], device=device),
                                      torch.tensor([1.0], device=device))

model = MLP(in_features=1, out_features=1, hidden_features=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 500

for epoch in range(epochs):
    
    batches = uniform.sample((batch_size, timesteps))
    batches_y = torch.sin(8 * np.pi * batches)
    for model_input, target in zip(batches, batches_y):
        optimizer.zero_grad()
        model_output = model(model_input)
        loss = F.mse_loss(model_output, target)
        loss.backward()
        optimizer.step()

eval_lenght = 1024
with torch.no_grad():
    test_input = uniform.sample((eval_lenght,))
    test_target = torch.sin(8 * np.pi * test_input)
    test_output = model(test_input)


# Plot the data in each partition in different colors:
plt.plot(test_input, test_output.cpu(), 'b.', label="Train")
plt.plot(test_input, test_target, 'r.', label="GT")
plt.legend()
plt.save('mlp1024_pytorch.png')

# set the model to inference mode
model.eval()
model_name = f"mlp{eval_lenght}.onnx"
# Export the model
torch.onnx.export(model,                       # model being run
                  test_input,                  # model input (or a tuple for multiple inputs)
                  model_name,                  # where to save the model (can be a file or file-like object)
                  export_params=True,          # store the trained parameter weights inside the model file
                  opset_version=8,             # the ONNX version to export the model to
                  # do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],     # the model's input names
                  output_names = ['output'])   # the model's output names