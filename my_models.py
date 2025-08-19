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
    
# =========================================================
# 3) Model (Swin-ViT backbone, FPN 넥, 7 keypoints)
# =========================================================

class PoseEstimationSwinFPN(nn.Module):
    def __init__(self, num_kp=7, backbone_name='swin_base_patch4_window16_512', pretrained=True):
        """
        Swin Transformer 백본과 FPN 넥을 사용하는 새로운 포즈 추정 모델
        'swin_base_patch4_window16_512'는 512x512 이미지로 사전 학습된 모델입니다.
        """
        super().__init__()
        self.num_kp = num_kp

        # 1. Swin Transformer 백본 로드
        # features_only=True로 설정하여 각 스테이지의 특징 맵을 모두 받아옵니다.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3) # 4개 스테이지의 출력을 모두 사용
        )
        
        # Swin-Base의 각 스테이지 출력 채널 수
        # (스테이지 0: 128, 1: 256, 2: 512, 3: 1024)
        backbone_channels = self.backbone.feature_info.channels()
        
        # FPN 넥에서 사용할 특징 맵의 채널 수
        fpn_channels = 256

        # 2. Feature Pyramid Network (FPN) 넥 정의
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        # 각 스테이지의 특징 맵을 fpn_channels로 맞춰주는 1x1 Conv (Lateral)
        for in_channels in backbone_channels:
            self.lateral_convs.append(nn.Conv2d(in_channels, fpn_channels, 1))
            self.fpn_convs.append(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1))

        # 3. 최종 Regression Head 정의
        # FPN의 최종 출력(고해상도)을 받아 키포인트를 예측합니다.
        self.regression_head = nn.Sequential(
            nn.Conv2d(fpn_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_kp * 3)
        )

    def forward(self, x):
        # 1. 백본을 통해 여러 스케일의 특징 맵 추출
        features = self.backbone(x) # [feat_s0, feat_s1, feat_s2, feat_s3]
        
        # 2. FPN 연산 수행
        # 2-1. 각 특징 맵의 채널을 fpn_channels로 통일 (Lateral connection)
        laterals = [
            lat_conv(features[i])
            for i, lat_conv in enumerate(self.lateral_convs)
        ]
        
        # 2-2. Top-down 경로: 고수준 특징을 저수준 특징과 결합
        # 가장 깊은 특징(laterals[3])부터 시작
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            # 이전 스테이지 특징 맵을 현재 스테이지 크기로 업샘플링하고 더해줌
            laterals[i-1] += F.interpolate(laterals[i], size=prev_shape, mode='nearest')
        
        # 2-3. 최종 특징 맵 생성 (3x3 Conv로 다듬어주기)
        fpn_outputs = [
            self.fpn_convs[i](laterals[i])
            for i in range(len(laterals))
        ]
        
        # 가장 해상도가 높은 fpn_outputs[0]를 헤드에 전달 (더 많은 스케일 사용도 가능)
        final_features = fpn_outputs[0]

        # 3. Regression Head를 통해 최종 3D 좌표 예측
        output = self.regression_head(final_features)
        
        return output.view(-1, self.num_kp, 3)