import os
import cv2
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# SegFormerForRobotArm import는 이제 필요 없습니다.
from my_utils import extract_joint_angles_from_row, angle_to_joint_coordinate

class RobotArmSegFKDataset(Dataset):
    def __init__(
        self,
        index_paths: List[str],  # 이제 'matched_index_with_roi.csv' 파일을 사용합니다.
        view_filter: Optional[str] = None,
        image_size: int = 512,
        img_key: str = "roi.path", # 이제 roi.path 컬럼을 읽습니다.
    ):
        self.image_size = image_size
        self.img_key = img_key

        rows = []
        for p in index_paths:
            df = pd.read_csv(p)
            rows.append(df)
        
        self.df = pd.concat(rows, ignore_index=True)
        
        # ROI를 찾지 못한 데이터를 제외합니다.
        self.df.dropna(subset=[self.img_key], inplace=True)
        
        if view_filter is not None:
            self.df = self.df[self.df.get("img.view", "") == view_filter]
        
        self.df = self.df.reset_index(drop=True)

        # 이제 모델 로딩 코드가 완전히 사라졌습니다.
        
        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 전처리된 ROI 이미지 경로를 바로 읽어옵니다.
        img_path = row[self.img_key]
        
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            # 혹시 파일이 없을 경우를 대비한 예외 처리
            raise FileNotFoundError(f"Preprocessed ROI image not found: {img_path}")
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # 더 이상 세그멘테이션, 마스크, 박스 계산이 필요 없습니다.
        
        transformed = self.transform(image=rgb)
        img_tensor = transformed['image']

        joint_angles = extract_joint_angles_from_row(row)
        if joint_angles is None:
            raise ValueError(f"No valid joint angles in row index={idx}")

        joints_xyz = angle_to_joint_coordinate(joint_angles)
        joints_tensor = torch.from_numpy(joints_xyz).float()

        sample = {
            "image": img_tensor,
            "joints_xyz": joints_tensor,
            "img_path": row["img.path"], # 원본 이미지 경로도 필요하면 전달
            "roi_path": img_path,
        }
        return sample