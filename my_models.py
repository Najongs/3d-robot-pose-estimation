import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
import timm

# =========================================================
# 1) SegFormer 모델 정의
# =========================================================
class SegFormerForRobotArm(nn.Module):
    def __init__(self, num_classes=2, model_name="nvidia/mit-b2"):
        super().__init__()
        self.num_classes = num_classes
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        total_params = sum(p.numel() for p in self.segformer.parameters())
        trainable_params = sum(p.numel() for p in self.segformer.parameters() if p.requires_grad)
        
    def forward(self, pixel_values):
        outputs = self.segformer(pixel_values=pixel_values)
        logits = outputs.logits
        upsampled_logits = F.interpolate(
            logits,
            size=pixel_values.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
        return upsampled_logits

# =========================================================
# 2) Model (HRNet-like with ViT backbone, 7 keypoints)
# =========================================================
def reshape_vit_output(x, patch_size=16, img_size=512):
    if x.ndim == 3 and x.shape[1] > 1:
        if x.shape[1] == (img_size // patch_size)**2 + 1:
            x = x[:, 1:]
        B, N, D = x.shape
        H_feat = W_feat = int(N**0.5)
        if H_feat * W_feat != N:
            raise ValueError(f"ViT output sequence length {N} is not a perfect square.")
        x = x.permute(0, 2, 1).reshape(B, D, H_feat, W_feat)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported ViT output shape for reshaping: {x.shape}")
    return x

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.down = (nn.Conv2d(in_c, out_c, 1, stride, bias=False)
                     if (stride != 1 or in_c != out_c) else None)
    def forward(self, x):
        res = self.down(x) if self.down else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)

class PoseEstimationHRViT(nn.Module):
    def __init__(self, num_kp=7, vit_model_name='vit_base_patch16_224', pretrained=True, img_size=512):
        super().__init__()
        self.num_kp = num_kp
        self.img_size = img_size
        self.vit_backbone = timm.create_model(
            vit_model_name,
            pretrained=True,
            num_classes=0,
            img_size=self.img_size  # <-- 이 인자를 추가!
        )
        print(f"Loaded pretrained ViT model: {vit_model_name}")
        vit_embed_dim = self.vit_backbone.embed_dim
        vit_patch_size = self.vit_backbone.patch_embed.patch_size[0]
        
        self.high_res_branch_conv = nn.Sequential(
            nn.Conv2d(vit_embed_dim, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            self._make_layer(256, 256, 2, 1)
        )
        self.low_res_branch_conv = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            self._make_layer(512, 512, 2, 1)
        )
        self.fuse_l_to_h = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_kp * 3)
        )
    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c))
        return nn.Sequential(*layers)
    def forward(self, x):
        vit_output = self.vit_backbone.forward_features(x)
        vit_spatial_features = reshape_vit_output(vit_output, self.vit_backbone.patch_embed.patch_size[0], self.img_size)
        high_res_feat = self.high_res_branch_conv(vit_spatial_features)
        low_res_feat = self.low_res_branch_conv(high_res_feat)
        high_res_fused = high_res_feat + self.fuse_l_to_h(low_res_feat)
        output = self.regression_head(high_res_fused)
        return output.view(-1, self.num_kp, 3)