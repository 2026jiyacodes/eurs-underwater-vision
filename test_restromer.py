import sys
from pathlib import Path

import torch

# 1. Find the official Restormer repository
RESTORMER_DIR = Path(__file__).resolve().parent / "Restormer"

# 2. Make its source code importable
sys.path.insert(0, str(RESTORMER_DIR))

# 3. Import the actual Restormer architecture
from basicsr.models.archs.restormer_arch import Restormer

# 4. Choose the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

# 5. Create the Restormer network
model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="WithBias",
    dual_pixel_task=False,
)

# 6. Move the model onto the selected device
model = model.to(device)

# 7. Create a fake RGB image
dummy_image = torch.rand(1, 3, 128, 128).to(device)

# 8. Run a forward pass
model.eval()

with torch.no_grad():
    restored_image = model(dummy_image)

# 9. Inspect the output
print("Input shape: ", dummy_image.shape)
print("Output shape:", restored_image.shape)
print("Output device:", restored_image.device)
print("Parameters:", sum(p.numel() for p in model.parameters()))
