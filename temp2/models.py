import torch
import torch.nn as nn
import torch.nn.functional as F

print("✅✅✅ SUCCESS: LOADED FINAL OPTIMIZED MODEL v4 (LOCAL ATTN) ✅✅✅")

# ==========================================
# SHARED COMPONENTS
# ==========================================
class ShallowMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(dim * 1.5)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# O(N) Local Attention Block (Depthwise Conv) for High Res (128x128)
class LocalBlock(nn.Module):
    def __init__(self, dim, shared_mlp):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim) # Channel-wise normalization
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.mlp = shared_mlp

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.dwconv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x + residual


# O(N²) Global Attention Block for Low Res (64x64) and ViT (256 tokens)
class GlobalBlock(nn.Module):
    def __init__(self, dim, shared_mlp, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.norm1 = nn.LayerNorm(dim)
        if use_attn:
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = shared_mlp

    def forward(self, x):
        # x is [B, L, C]
        if self.use_attn:
            attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
            x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GatedFeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, vit_feat, restormer_feat):
        gate_weights = self.gate_net(restormer_feat)
        return vit_feat + (restormer_feat * gate_weights)


# ==========================================
# MICRO-RESTORMER
# ==========================================
class MicroRestormer(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.stem = nn.Conv2d(4, dim, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1) # 256 -> 128
        
        self.shared_mlp_high = ShallowMLP(dim)
        # Use LocalBlock (O(N)) for 128x128 to save massive GFLOPs
        self.enc1 = nn.Sequential(*[LocalBlock(dim, self.shared_mlp_high) for _ in range(2)])
        
        self.down2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1) # 128 -> 64
        self.shared_mlp_low = ShallowMLP(dim*2)
        # Use GlobalBlock (O(N²)) for 64x64 (only 4096 tokens, very cheap)
        self.enc2 = nn.Sequential(*[GlobalBlock(dim*2, self.shared_mlp_low, use_attn=True) for _ in range(2)])
        
        self.up1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        self.up2 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.stem(x)           # [B, 32, 256, 256]
        x = self.down1(x)          # [B, 32, 128, 128]
        x = self.enc1(x)           # Local attention (O(N)), super cheap
        
        x = self.down2(x)          # [B, 64, 64, 64]
        
        # Prepare for GlobalBlock (needs [B, L, C])
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.enc2(x_flat) # Global attention at 64x64 (4096 tokens, cheap)
        restormer_feat = x_flat.transpose(1, 2).view(B, C, H, W)
        
        x = self.up1(restormer_feat)   # [B, 32, 128, 128]
        x = self.dec1(x)               # Conv only
        x = self.up2(x)                # [B, 32, 256, 256]
        x = self.dec2(x)               # Conv only
        
        return self.out_conv(x), restormer_feat


# ==========================================
# MICRO-ViT
# ==========================================
class MicroViT(nn.Module):
    def __init__(self, dim=64, patch_size=16, num_classes=3, restormer_dim=64):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.shared_mlp = ShallowMLP(dim)
        
        # 256 tokens. O(N²) is 256² * 64 = 4M MACs. Completely negligible.
        self.layers = nn.ModuleList([
            GlobalBlock(dim, self.shared_mlp, use_attn=True),
            GlobalBlock(dim, self.shared_mlp, use_attn=False),
            GlobalBlock(dim, self.shared_mlp, use_attn=True),
            GlobalBlock(dim, self.shared_mlp, use_attn=False),
        ])
        self.norm = nn.LayerNorm(dim)
        
        self.upsample_to_half = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.fusion_gate = GatedFeatureFusion(dim=restormer_dim)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.seg_head = nn.Conv2d(restormer_dim, num_classes, kernel_size=1)

    def forward(self, x, restormer_feat):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        vit_feat_half = self.upsample_to_half(x)
        fused_feat = self.fusion_gate(vit_feat_half, restormer_feat)
        full_res = self.final_upsample(fused_feat)
        
        return self.seg_head(full_res)


# ==========================================
# JOINT PIPELINE
# ==========================================
class JointPipeline(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.restorer = MicroRestormer(dim=32)
        self.segmenter = MicroViT(dim=64, patch_size=16, num_classes=num_classes, restormer_dim=64)

    def forward(self, x_4ch):
        restored_img, restormer_feat = self.restorer(x_4ch)
        seg_logits = self.segmenter(restored_img, restormer_feat)
        return restored_img, seg_logits