import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Selected device:", device)

x = torch.randn(2, 3, 64, 64, device=device)
print("Test tensor:", x.shape, x.device)
