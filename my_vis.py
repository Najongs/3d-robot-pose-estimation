# visualize.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 기존 프로젝트의 모델과 데이터셋 클래스를 가져옵니다.
from my_models import PoseEstimationSwinFPN
from my_datasets import RobotArmSegFKDataset

# ======================= 설정 =======================
# 학습된 최고 성능의 모델 가중치 경로
MODEL_PATH = "checkpoints_ddp_v2/best_model_checkpoint.pt" 

# 시각화에 사용할 CSV 파일 경로
VAL_CSV_PATH = os.path.join(
    "/home/najo/NAS/DIP/datasets/Fr5_intertek_dataset",
    "Fr5_intertek_7th_250526/matched_index_with_roi.csv"
)

# 시각화할 샘플 개수
NUM_SAMPLES_TO_VIZ = 10 # 저장할 이미지 개수를 늘려도 좋습니다.

# --- 수정: 결과 이미지를 저장할 폴더 이름 ---
RESULT_DIR = "result"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
# ====================================================

def load_model(model_path, device):
    """DDP로 학습된 모델을 불러옵니다."""
    model = PoseEstimationSwinFPN(num_kp=7, img_size=IMG_SIZE)
    state_dict = torch.load(model_path, map_location=device)
    
    if all(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def plot_3d_pose(ax, points_3d, color, label, connections):
    """3D 스캐터 플롯에 로봇팔 스켈레톤을 그립니다."""
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=color, marker='o', label=label)
    
    for start_idx, end_idx in connections:
        ax.plot(
            [points_3d[start_idx, 0], points_3d[end_idx, 0]],
            [points_3d[start_idx, 1], points_3d[end_idx, 1]],
            [points_3d[start_idx, 2], points_3d[end_idx, 2]],
            c=color
        )

def main():
    # --- 수정: 결과 폴더 생성 ---
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"결과 이미지는 '{RESULT_DIR}' 폴더에 저장됩니다.")

    model = load_model(MODEL_PATH, DEVICE)
    dataset = RobotArmSegFKDataset(
        index_paths=[VAL_CSV_PATH],
        image_size=IMG_SIZE
    )
    
    skeleton_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)
    ]

    indices = np.random.choice(len(dataset), NUM_SAMPLES_TO_VIZ, replace=False)
    
    for i in indices:
        sample = dataset[i]
        image_tensor = sample['image'].unsqueeze(0).to(DEVICE)
        gt_joints_xyz = sample['joints_xyz'].numpy()
        roi_path = sample['roi_path']

        with torch.no_grad():
            pred_joints_xyz = model(image_tensor).squeeze(0).cpu().numpy()

        fig = plt.figure(figsize=(12, 6))
        
        ax_img = fig.add_subplot(1, 2, 1)
        roi_image = cv2.imread(roi_path)
        ax_img.imshow(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
        ax_img.set_title(f"Input ROI Image (Sample #{i})")
        ax_img.axis('off')

        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
        plot_3d_pose(ax_3d, gt_joints_xyz, 'blue', 'Ground Truth', skeleton_connections)
        plot_3d_pose(ax_3d, pred_joints_xyz, 'red', 'Prediction', skeleton_connections)
        
        ax_3d.set_xlabel('X coordinate')
        ax_3d.set_ylabel('Y coordinate')
        ax_3d.set_zlabel('Z coordinate')
        ax_3d.set_title("3D Pose Estimation Result")
        ax_3d.legend()
        ax_3d.set_aspect('equal')

        plt.tight_layout()
        
        # --- 수정: 파일명 생성 및 저장 ---
        base_filename = os.path.basename(roi_path)
        save_path = os.path.join(RESULT_DIR, f"result_{base_filename}")
        plt.savefig(save_path)
        print(f"Saved result to {save_path}")
        
        # --- 수정: 메모리 관리를 위해 plot 닫기 ---
        plt.close(fig)

if __name__ == "__main__":
    main()