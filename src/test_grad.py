import torch
weights = torch.nn.Parameter(torch.ones(3, 4), requires_grad=True)
inputs = torch.nn.Parameter(torch.ones(4), requires_grad=True)
outputs = torch.matmul(weights, inputs)
print(torch.autograd.grad(outputs, inputs, torch.ones(3)))