import torch
import torch.nn as nn
import torch.nn.functional as F

print("✅✅✅ SUCCESS: LOADED FINAL OPTIMIZED MODEL v5 (PRUNING + DINO DISTILL) ✅✅✅")

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


class LocalBlock(nn.Module):
    def __init__(self, dim, shared_mlp):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
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
        if restormer_feat.shape[2:] != vit_feat.shape[2:]:
            restormer_feat = F.interpolate(
                restormer_feat, 
                size=vit_feat.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        gate_weights = self.gate_net(restormer_feat)
        return vit_feat + (restormer_feat * gate_weights)


# ==========================================
# MICRO-RESTORMER (Unchanged)
# ==========================================
class MicroRestormer(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.stem = nn.Conv2d(4, dim, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1)
        
        self.shared_mlp_high = ShallowMLP(dim)
        self.enc1 = nn.Sequential(*[LocalBlock(dim, self.shared_mlp_high) for _ in range(2)])
        
        self.down2 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1)
        self.shared_mlp_low = ShallowMLP(dim*2)
        self.enc2 = nn.Sequential(*[GlobalBlock(dim*2, self.shared_mlp_low, use_attn=True) for _ in range(2)])
        
        self.up1 = nn.ConvTranspose2d(dim*2, dim, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.GELU())
        
        self.up2 = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1), nn.GELU())
        self.out_conv = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.enc1(x)
        
        x = self.down2(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = self.enc2(x_flat)
        restormer_feat = x_flat.transpose(1, 2).view(B, C, H, W)
        
        x = self.up1(restormer_feat)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        
        return self.out_conv(x), restormer_feat


# ==========================================
# MICRO-ViT (UPDATED: Token Pruning + DINO Projection)
# ==========================================
class MicroViT(nn.Module):
    def __init__(self, dim=64, patch_size=16, num_classes=3, restormer_dim=64, dino_dim=384):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.shared_mlp = ShallowMLP(dim)
        
        self.layers = nn.ModuleList([
            GlobalBlock(dim, self.shared_mlp, use_attn=True),   # Layer 1
            GlobalBlock(dim, self.shared_mlp, use_attn=False),  # Layer 2
            GlobalBlock(dim, self.shared_mlp, use_attn=True),   # Layer 3
            GlobalBlock(dim, self.shared_mlp, use_attn=False),  # Layer 4
        ])
        self.norm = nn.LayerNorm(dim)
        
        # Projection head to match DINOv2 ViT-S feature dimension (384) for distillation
        self.dino_proj = nn.Linear(dim, dino_dim)
        
        self.upsample_to_half = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.fusion_gate = GatedFeatureFusion(dim=restormer_dim)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.seg_head = nn.Conv2d(restormer_dim, num_classes, kernel_size=1)

    def forward(self, x, restormer_feat):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, 256, 64]
        
        # 1. Layer 1 (Full resolution tokens)
        x = self.layers[0](x)
        
        # 2. DYNAMIC TOKEN PRUNING (The GFLOP Saver)
        # Calculate variance across channel dimension. Uniform water = low variance, objects = high variance.
        scores = x.var(dim=-1)  # [B, 256]
        k = int(0.7 * x.shape[1]) # Keep exactly 70% of tokens
        
        # Get indices of top-k most informative tokens
        _, indices = torch.topk(scores, k, dim=1) # [B, k]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]) # [B, k, 64]
        
        # Gather kept tokens
        x_pruned = torch.gather(x, 1, indices_expanded) # [B, k, 64]
        
        # 3. Layers 2-4 process ONLY the 70% kept tokens (Massive GFLOP reduction)
        for i in range(1, len(self.layers)):
            x_pruned = self.layers[i](x_pruned)
            
        # 4. Unprune: Scatter back to original [B, 256, 64] grid for segmentation head
        # Dropped tokens are filled with zeros, which bilinear upsampling handles gracefully.
        x_full = torch.zeros_like(x)
        x_full = torch.scatter(x_full, 1, indices_expanded, x_pruned)
        
        # Save projected tokens for DINO Distillation Loss
        vit_tokens_for_distill = self.dino_proj(x_full) # [B, 256, 384]
        
        # 5. Continue to Segmentation Head
        x_full = self.norm(x_full)
        x_full = x_full.transpose(1, 2).view(B, C, H, W)
        
        vit_feat_half = self.upsample_to_half(x_full)
        fused_feat = self.fusion_gate(vit_feat_half, restormer_feat)
        full_res = self.final_upsample(fused_feat)
        
        return self.seg_head(full_res), vit_tokens_for_distill


# ==========================================
# JOINT PIPELINE (UPDATED)
# ==========================================
class JointPipeline(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.restorer = MicroRestormer(dim=32)
        self.segmenter = MicroViT(dim=64, patch_size=16, num_classes=num_classes, restormer_dim=64)

    def forward(self, x_4ch):
        restored_img, restormer_feat = self.restorer(x_4ch)
        # Extract 3-channel RGB from restored for the ViT (ViT expects 3 channels)
        seg_logits, vit_tokens = self.segmenter(restored_img[:, :3, :, :], restormer_feat)
        return restored_img, seg_logits, vit_tokens