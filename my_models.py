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
    def __init__(self, num_kp=7, img_size=512, backbone_name='swin_base_patch4_window12_384', pretrained=True):
        super().__init__()
        self.num_kp = num_kp

        # 1. 백본 로드: FPN에 주로 사용되는 3개의 고수준 특징 맵만 사용 (out_indices=(1, 2, 3))
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3), # 더 안정적인 FPN을 위해 3개의 스테이지만 사용
            img_size=img_size
        )
        
        # Swin-Base의 각 스테이지 출력 채널 수 (예: [256, 512, 1024])
        backbone_channels = self.backbone.feature_info.channels()
        
        # FPN 넥에서 사용할 공통 채널 수
        fpn_channels = 256

        # 2. FPN 넥 정의 (더 명확한 구조로 수정)
        self.lateral_convs = nn.ModuleList() # Top-down 경로의 1x1 Conv
        self.output_convs = nn.ModuleList()  # 최종 출력을 다듬는 3x3 Conv

        # Backbone의 각 스테이지 출력을 fpn_channels로 맞춰주는 Lateral Conv 생성
        for in_channels in reversed(backbone_channels):
            self.lateral_convs.append(nn.Conv2d(in_channels, fpn_channels, 1))
        
        # 각 FPN 레벨의 출력을 다듬어주는 Output Conv 생성
        for _ in range(len(backbone_channels)):
             self.output_convs.append(nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1))

        # 3. Regression Head 정의
        self.regression_head = nn.Sequential(
            nn.Conv2d(fpn_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_kp * 3)
        )

    def forward(self, x):
        # 1. 백본에서 특징 맵 추출
        features = self.backbone(x)

        # --- !! 해결책: 메모리 형식을 Channels-First (B, C, H, W)로 변환 !! ---
        # 백본 출력이 channels-last 형식일 수 있으므로, permute로 순서를 바꿔줍니다.
        features = [f.permute(0, 3, 1, 2) for f in features]
        # --------------------------------------------------------------------

        features = list(reversed(features)) # 깊은 것부터 처리하기 위해 순서 뒤집기

        # 2. FPN Top-down 경로 연산
        lateral_outputs = []
        prev_feature = None

        for i, feature in enumerate(features):
            current_feature = self.lateral_convs[i](feature)
            if prev_feature is not None:
                current_feature += F.interpolate(prev_feature, size=current_feature.shape[2:], mode='nearest')
            
            lateral_outputs.append(current_feature)
            prev_feature = current_feature
        
        # 3. FPN Output Conv 연산
        fpn_outputs = [
            self.output_convs[i](feature)
            for i, feature in enumerate(reversed(lateral_outputs))
        ]
        
        final_feature = fpn_outputs[0]

        # 4. Regression Head로 최종 3D 좌표 예측
        output = self.regression_head(final_feature)
        
        return output.view(-1, self.num_kp, 3)