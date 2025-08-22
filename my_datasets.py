# in my_datasets.py

import os
import cv2
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# my_utils.py에서 필요한 함수들을 모두 import한다고 가정합니다.
from my_utils import load_all_intrinsics, load_all_extrinsics, get_camera_parameters
from my_utils import extract_joint_angles_from_row, angle_to_joint_coordinate, transform_robot_to_camera_coords

class RobotArmPoseDataset(Dataset):
    def __init__(
        self,
        index_paths: List[str],
        intrinsics_path: str,  # 단일 경로로 변경
        extrinsics_path: str,  # 단일 경로로 변경
        serial_map: Dict[str, str],
        view_filter: Optional[str] = None,
        image_size: int = 512,
    ):
        self.image_size = image_size
        self.serial_map = serial_map

        # 1. 모든 파라미터를 한 번에 미리 로드 (세션 구분 없음)
        print("Loading centralized calibration files...")
        self.intrinsics = load_all_intrinsics(intrinsics_path)
        self.extrinsics = load_all_extrinsics(extrinsics_path)
        if not self.intrinsics or not self.extrinsics:
            raise RuntimeError(f"Failed to load calibration from paths:\nIntrinsics: {intrinsics_path}\nExtrinsics: {extrinsics_path}")
        print("✅ Calibrations loaded.")
            
        # 2. CSV 파일 로드 (세션 컬럼 추가 로직 제거)
        all_dfs = [pd.read_csv(csv_path) for csv_path in index_paths]
        self.df = pd.concat(all_dfs, ignore_index=True)
        self.df.dropna(subset=['roi.path', 'img.path'], inplace=True)
        if view_filter is not None:
            self.df = self.df[self.df["img.view"] == view_filter]
        self.df = self.df.reset_index(drop=True)

        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        roi_path = row["roi.path"]
        bgr = cv2.imread(roi_path, cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(f"ROI image not found: {roi_path}")
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR_RGB)
        img_tensor = self.transform(image=bgr)['image']

        joint_angles = extract_joint_angles_from_row(row)
        if joint_angles is None: raise ValueError(f"No valid joint angles in row index={idx}")
        joints_robot_xyz = angle_to_joint_coordinate(joint_angles)

        original_img_filename = os.path.basename(row["img.path"])
        
        # 3. 미리 로드된 단일 파라미터 셋을 사용
        cam_params = get_camera_parameters(
            original_img_filename,
            self.intrinsics,
            self.extrinsics,
            self.serial_map
        )
        if cam_params is None: raise RuntimeError(f"Cam params not found for {original_img_filename}")

        joints_camera_xyz = transform_robot_to_camera_coords(joints_robot_xyz, cam_params)
        joints_tensor = torch.from_numpy(joints_camera_xyz).float()

        return {
            "image": img_tensor,
            "joints_camera_xyz": joints_tensor,
            "img_path": row["img.path"],
        }