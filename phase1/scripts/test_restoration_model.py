
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.restoration_model import UnderwaterRestorationNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UnderwaterRestorationNet().to(device)
    model.eval()

    # Simulate a batch of two RGB images, 384 x 512.
    dummy_input = torch.rand(2, 3, 384, 512).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print("Input shape:", tuple(dummy_input.shape))
    print("Output shape:", tuple(output.shape))
    print("Output minimum:", output.min().item())
    print("Output maximum:", output.max().item())

    assert output.shape == dummy_input.shape, "Shape mismatch!"
    assert output.min().item() >= 0.0, "Output below 0!"
    assert output.max().item() <= 1.0, "Output above 1!"

    print("Restoration model test passed!")


if __name__ == "__main__":
    main()
