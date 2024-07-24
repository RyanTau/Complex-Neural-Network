import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn import _reduction as _Reduction


real = torch.tensor(0).float()
imaginary = torch.tensor(3).float()

cmplx_tensor = torch.complex(real, imaginary)
cmplx_tensor2 = torch.complex(real, imaginary)
print(cmplx_tensor)



# add tensors
tensor_add = torch.add(cmplx_tensor, cmplx_tensor2)

# multiply tensors
tensor_mul = torch.mul(cmplx_tensor, cmplx_tensor2)

# Define the amount of synthetic data samples
n_samples = 1000

# Create synthetic input data
X = torch.randn(n_samples, 1, dtype=torch.float)
X_complex = torch.complex(X, X)
Y = X_complex**2  # square numbers
# Y = (X_complex.real > 0.5).float()
# Y = F.one_hot(Y.to(torch.int64))

class ComplexLinear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=torch.cfloat) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize weights and bias with complex numbers
        with torch.no_grad():
            real_part = torch.randn_like(self.weight, dtype=torch.float)
            imag_part = torch.randn_like(self.weight, dtype=torch.float)
            self.weight.copy_(torch.complex(real_part, imag_part))
            if self.bias is not None:
                real_part_bias = torch.randn_like(self.bias, dtype=torch.float)
                imag_part_bias = torch.randn_like(self.bias, dtype=torch.float)
                self.bias.copy_(torch.complex(real_part_bias, imag_part_bias))

    def forward(self, input: Tensor) -> Tensor:
        # print(input.dtype, self.weight.dtype, self.bias.dtype, " WEIGHTS ")
        # print(input.shape, self.weight.shape, "SHAPES")
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


def complex_relu(z):
    return torch.complex(F.relu(z.real), F.relu(z.imag))

x = 0
# Definr model with multiple linear layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = ComplexLinear(1, 128)   # Input layer
        self.fc2 = ComplexLinear(128, 64)   # Input layer
        self.fc3 = ComplexLinear(64, 1) # Hidden layer
        # self.fc4 = ComplexLinear(32, 1) # Hidden layer
        # self.fc3 = ComplexLinear(512, 1)   # Output layer

        # self.set_weights()
    def forward(self, x):
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = complex_relu(x)
        x = self.fc3(x)
        # x = complex_relu(x)
        # x = self.fc4(x)
        return x
    
    def set_weights(self):
        print(self.fc1.weight)

class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class ComplexMSELoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.complex_mse_loss(input, target)
    
    def complex_mse_loss(self, output, target):
        return (0.5*(output - target)**2).mean(dtype=torch.complex64)
    
class ComplexDistanceLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return torch.abs(input - target).mean(dtype=torch.complex64)

    
class ComplexME(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.complex_me_loss(input, target)
    
    def complex_me_loss(self, output, target):
        return (output - target).mean(dtype=torch.complex64)
    



model = Net()
# torch.nn.MSELoss
# Define mean squared error loss function
# Define SGD optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = ComplexDistanceLoss()

epochs = 10000
for epoch in range(epochs):
    outputs = model(X_complex)
    loss = criterion(outputs, Y)
    # print(X_complex[0], " X")
    # print(model(X_complex[0]), " Y")
    optimizer.zero_grad()
    loss.backward()
    
    # Print gradients
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Gradient of {name} during epoch {epoch + 1}: {param.grad}")
    optimizer.step()

    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')

