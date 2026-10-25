"""
Tiny multi-task model for EURS: one shared encoder, two heads.
Outputs: restored_image (1,3,H,W) in [0,1], seg_logits (1,5,H,W).
Target: < 1.0M params, < 12 GFLOPs at the chosen static H,W.
NAFNet-style blocks (no attention, cheap, export cleanly to ONNX).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, c, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        dt = x.dtype
        x = x.float()  # norm in fp32: avoids fp16 overflow/NaN under AMP
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        return ((x - mu) / torch.sqrt(var + 1e-6) * self.w + self.b).to(dt)


class WhiteBalance(nn.Module):
    """Learnable per-channel gain + bias. Removes the blue-green cast cheaply (6 params)."""
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, 3, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x):
        return torch.clamp(x * self.gain + self.bias, 0.0, 1.0)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1 = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, 2 * c, 1)
        self.dw = nn.Conv2d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1))
        self.pw2 = nn.Conv2d(c, c, 1)
        self.norm2 = LayerNorm2d(c)
        self.ffn1 = nn.Conv2d(c, 2 * c, 1)
        self.ffn2 = nn.Conv2d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = self.gate(self.dw(self.pw1(self.norm1(x))))
        y = self.pw2(y * self.sca(y))
        x = x + y * self.beta
        y = self.ffn2(self.gate(self.ffn1(self.norm2(x))))
        return x + y * self.gamma


def stage(c, n):
    return nn.Sequential(*[NAFBlock(c) for _ in range(n)])


class Up(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_in * 2, 1, bias=False)  # -> c_in/2 after shuffle
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.conv(x))


class Decoder(nn.Module):
    def __init__(self, widths, dec_blocks):
        super().__init__()
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.ups.append(Up(widths[i]))
            self.blocks.append(stage(widths[i - 1], dec_blocks[i - 1]))

    def forward(self, x, skips):
        for up, blk, s in zip(self.ups, self.blocks, reversed(skips)):
            x = blk(up(x) + s)
        return x


class TinyEURS(nn.Module):
    def __init__(self, widths=(16, 32, 64, 128), enc_blocks=(1, 1, 2, 4),
                 dec_blocks=(1, 1, 1), n_classes=5):
        super().__init__()
        self.wb = WhiteBalance()
        self.intro = nn.Conv2d(3, widths[0], 3, padding=1)
        self.enc = nn.ModuleList([stage(w, n) for w, n in zip(widths, enc_blocks)])
        self.down = nn.ModuleList(
            [nn.Conv2d(widths[i], widths[i + 1], 2, stride=2) for i in range(len(widths) - 1)])
        self.rest_dec = Decoder(widths, dec_blocks)
        self.seg_dec = Decoder(widths, dec_blocks)
        self.rest_out = nn.Conv2d(widths[0], 3, 3, padding=1)
        self.seg_out = nn.Conv2d(widths[0], n_classes, 3, padding=1)

    def forward(self, inp):
        wb = self.wb(inp)
        x = self.intro(wb)
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        delta = 0.5 * torch.tanh(self.rest_out(self.rest_dec(x, skips)))  # bounded correction
        restored = torch.clamp(wb + delta, 0.0, 1.0)
        seg = self.seg_out(self.seg_dec(x, skips))
        return restored, seg


if __name__ == "__main__":
    # Quick check: parameter count + output shapes. Safe to run in a notebook cell.
    H, W = 384, 512
    m = TinyEURS().eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}  ({n/1e6:.3f}M)")
    with torch.no_grad():
        r, s = m(torch.rand(1, 3, H, W))
    print("restored_image", tuple(r.shape), "| seg_logits", tuple(s.shape))
