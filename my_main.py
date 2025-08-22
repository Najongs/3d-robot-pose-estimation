import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# DDP 관련 모듈 import
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

import wandb

## --- CORRECTED ---
# 단순화된 데이터셋을 import 합니다.
from my_datasets import RobotArmPoseDataset
from my_models import PoseEstimationHRViT, PoseEstimationSwinFPN

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
ROOT = "/home/najo/NAS/DIP/datasets/Fr5_intertek_dataset"

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

# =========================================================
# 1) 학습 및 검증 루프 (W&B 연동 버전)
# =========================================================
def train_and_validate(local_rank: int, model: nn.Module, train_dataset: Dataset, val_dataset: Dataset, config: dict):
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], persistent_workers=True, pin_memory=True, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True, sampler=val_sampler
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    criterion = nn.SmoothL1Loss(reduction='mean')
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    if local_rank == 0:
        os.makedirs(config['save_path'], exist_ok=True)
        print(f"Starting training for {config['epochs']} epochs...")

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", disable=(local_rank != 0))
        
        for batch in train_progress_bar:
            images = batch['image'].to(local_rank)
            ## --- CORRECTED ---
            # 데이터셋에서 반환하는 레이블 키를 'joints_camera_xyz'로 변경
            labels = batch['joints_camera_xyz'].to(local_rank)
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) # 올바른 레이블 사용
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            
        scheduler.step()

        train_loss_tensor = torch.tensor([train_loss]).to(local_rank)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_train_loss = train_loss_tensor.item() / len(train_sampler.dataset)

        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", disable=(local_rank != 0))
        with torch.no_grad():
            for batch in val_progress_bar:
                images = batch['image'].to(local_rank)
                ## --- CORRECTED ---
                # 검증 데이터의 레이블 키도 'joints_camera_xyz'로 변경
                labels = batch['joints_camera_xyz'].to(local_rank)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels) # 올바른 레이블 사용
                    val_loss += loss.item() * images.size(0)
        
        val_loss_tensor = torch.tensor([val_loss]).to(local_rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_val_loss = val_loss_tensor.item() / len(val_sampler.dataset)

        if local_rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.6f}")
            
            wandb.log({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "learning_rate": current_lr,
                "epoch": epoch + 1
            })

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                print(f"Validation loss improved to {best_val_loss:.4f}. Saving best model...")
                torch.save(model.module.state_dict(), os.path.join(config['save_path'], "best_model_checkpoint.pt"))
                
    if local_rank == 0:
        print("Training finished.")
        wandb.finish()
    cleanup_ddp()

# =========================================================
# 2) 실행 메인 함수
# =========================================================
def main():
    local_rank = setup_ddp()
    
    config = {
        'epochs': 50,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_workers': 24,
        'save_path': "checkpoints_ddp_v2",
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
    }

    if local_rank == 0:
        wandb.init(
            project="FR5_ROBOT_3D_POSE",
            config=config,
            name=f"ddp_run_{config['batch_size']}_{config['learning_rate']}"
        )
        
    ## --- CORRECTED ---
    # 1. 중앙화된 파라미터 경로를 명시적으로 정의
    CAMERA_SERIALS = {"top": "30779426", "right": "34850673", "left": "38007749"}
    INTRINSICS_PATH = "/home/najo/NAS/DIP/3d-robot-pose-estimation/camera_conf"
    EXTRINSICS_PATH = "/home/najo/NAS/DIP/3d-robot-pose-estimation/ArUco_result_output/fr5_aruco_pose_summary.json"

    # 2. 학습/검증 CSV 파일 목록 정의
    train_sessions = [f'Fr5_intertek_{i}th_250526' for i in range(1, 7)]
    val_sessions = ['Fr5_intertek_7th_250526']
    
    train_index_files = [os.path.join(ROOT, s, 'matched_index_with_roi.csv') for s in train_sessions]
    val_index_files = [os.path.join(ROOT, s, 'matched_index_with_roi.csv') for s in val_sessions]
    
    # 3. 단순화된 RobotArmPoseDataset으로 데이터셋 생성
    IMG_SIZE = 512
    train_dataset = RobotArmPoseDataset(
        index_paths=train_index_files,
        intrinsics_path=INTRINSICS_PATH,
        extrinsics_path=EXTRINSICS_PATH,
        serial_map=CAMERA_SERIALS,
        image_size=IMG_SIZE
    )
    val_dataset = RobotArmPoseDataset(
        index_paths=val_index_files,
        intrinsics_path=INTRINSICS_PATH,
        extrinsics_path=EXTRINSICS_PATH,
        serial_map=CAMERA_SERIALS,
        image_size=IMG_SIZE
    )
    
    model = PoseEstimationSwinFPN(num_kp=7, pretrained=True, img_size=IMG_SIZE).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if local_rank == 0:
        wandb.watch(model, log="all", log_freq=100)

    train_and_validate(
        local_rank=local_rank,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )

if __name__ == "__main__":
    print("메인 학습 스크립트를 시작합니다.")
    main()