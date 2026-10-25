import torch
import torch.nn as nn
import torch.nn.functional as F

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


class MicroTransformerBlock(nn.Module):
    def __init__(self, dim, shared_mlp):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = shared_mlp

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        attn_out = self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat), need_weights=False)[0]
        x_flat = x_flat + attn_out
        mlp_out = self.mlp(self.norm2(x_flat))
        x_flat = x_flat + mlp_out
        return x_flat.transpose(1, 2).view(B, C, H, W)


class MicroViTBlock(nn.Module):
    def __init__(self, dim, shared_mlp, has_attention=True):
        super().__init__()
        self.has_attention = has_attention
        if has_attention:
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = shared_mlp

    def forward(self, x):
        if self.has_attention:
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
# MICRO-RESTORMER (Dim reduced to 32 for massive GFLOP safety)
# ==========================================
class MicroRestormer(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.stem = nn.Conv2d(4, dim, kernel_size=3, padding=1)
        self.shared_mlp = ShallowMLP(dim)
        
        self.enc1 = nn.Sequential(*[MicroTransformerBlock(dim, self.shared_mlp) for _ in range(2)])
        self.down = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1)
        
        # enc2 operates at 1/2 resolution (128x128 for 256x256 input)
        self.enc2 = nn.Sequential(*[MicroTransformerBlock(dim*2, ShallowMLP(dim*2)) for _ in range(2)])
        
        self.up = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(*[MicroTransformerBlock(dim, self.shared_mlp) for _ in range(2)])
        self.out_conv = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.enc1(x)
        x = self.down(x)
        
        restormer_feat = self.enc2(x) # Save 1/2 res feature for the gate
        
        x = self.up(restormer_feat)
        x = self.dec1(x)
        return self.out_conv(x), restormer_feat


# ==========================================
# MICRO-ViT (Dim reduced to 64, Profiler-Friendly Upsampling)
# ==========================================
class MicroViT(nn.Module):
    def __init__(self, dim=64, patch_size=16, num_classes=3, restormer_dim=64):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.shared_mlp = ShallowMLP(dim)
        
        self.layers = nn.ModuleList([
            MicroViTBlock(dim, self.shared_mlp, has_attention=True),
            MicroViTBlock(dim, self.shared_mlp, has_attention=False),
            MicroViTBlock(dim, self.shared_mlp, has_attention=True),
            MicroViTBlock(dim, self.shared_mlp, has_attention=False),
        ])
        self.norm = nn.LayerNorm(dim)
        
        # Profiler-friendly upsampling (thop calculates this correctly as near-zero)
        self.upsample_to_half = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.fusion_gate = GatedFeatureFusion(dim=restormer_dim)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 1x1 conv is extremely cheap and profiler-friendly
        self.seg_head = nn.Conv2d(restormer_dim, num_classes, kernel_size=1)

    def forward(self, x, restormer_feat):
        x = self.patch_embed(x) # [B, dim, 16, 16]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, 256, dim]
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W) # [B, dim, 16, 16]
        
        # Upsample to 128x128 to match restormer_feat
        vit_feat_half = self.upsample_to_half(x)
        
        # Fuse with Restormer's clean edge features
        fused_feat = self.fusion_gate(vit_feat_half, restormer_feat)
        
        # Upsample to 256x256 and project to classes
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