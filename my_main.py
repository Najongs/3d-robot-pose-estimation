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

## --- W&B 연동 1: wandb import ---
import wandb

from my_datasets import RobotArmSegFKDataset
from my_models import PoseEstimationHRViT

# ... (setup_ddp, cleanup_ddp 함수 및 기타 설정은 모두 동일) ...
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
    # ... (DataLoader, Optimizer, Scheduler, Scaler 설정은 모두 동일) ...
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
    scaler = torch.amp.GradScaler()
    
    best_val_loss = float('inf')
    if local_rank == 0:
        os.makedirs(config['save_path'], exist_ok=True)
        print(f"Starting training for {config['epochs']} epochs...")

    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)
        # ... (train_one_epoch 로직은 모두 동일) ...
        model.train()
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]", disable=(local_rank != 0))
        
        for batch in train_progress_bar:
            images = batch['image'].to(local_rank)
            joints_xyz = batch['joints_xyz'].to(local_rank)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, joints_xyz)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)
            
        scheduler.step()

        # ... (validate_one_epoch 로직 및 손실 동기화는 모두 동일) ...
        train_loss_tensor = torch.tensor([train_loss]).to(local_rank)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_train_loss = train_loss_tensor.item() / len(train_sampler.dataset)

        model.eval()
        val_loss = 0.0
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]", disable=(local_rank != 0))
        with torch.no_grad():
            for batch in val_progress_bar:
                with torch.amp.autocast(device_type='cuda'):
                    images = batch['image'].to(local_rank)
                    joints_xyz = batch['joints_xyz'].to(local_rank)
                    outputs = model(images)
                    loss = criterion(outputs, joints_xyz)
                    val_loss += loss.item() * images.size(0)
        
        val_loss_tensor = torch.tensor([val_loss]).to(local_rank)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_val_loss = val_loss_tensor.item() / len(val_sampler.dataset)

        if local_rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.6f}")
            
            ## --- W&B 연동 4: wandb.log로 지표 기록 ---
            wandb.log({
                "train_loss": epoch_train_loss,
                "val_loss": epoch_val_loss,
                "learning_rate": current_lr,
                "epoch": epoch + 1
            })

            # ... (체크포인트 저장 로직은 동일) ...
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                # ... (모델 저장 코드) ...

    if local_rank == 0:
        print("Training finished.")
        ## --- W&B 연동 5: wandb 실행 종료 ---
        wandb.finish()
    cleanup_ddp()

# =========================================================
# 2) 실행 메인 함수
# =========================================================
def main():
    local_rank = setup_ddp()
    
    config = {
        'epochs': 50,
        'batch_size': 54,
        'learning_rate': 2e-4,
        'num_workers': 24,
        'save_path': "checkpoints_ddp_v2",
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
    }

    ## --- W&B 연동 2: wandb.init 초기화 (메인 프로세스에서만) ---
    if local_rank == 0:
        wandb.init(
            project="FR5_ROBOT_3D_POSE", # W&B 프로젝트 이름
            config=config,                    # 학습 설정값들을 저장
            name=f"ddp_run_{config['batch_size']}_{config['learning_rate']}" # 실행 이름
        )

    train_index_files = [os.path.join(ROOT, 'Fr5_intertek_1st_250526/matched_index_with_roi.csv'),
                        os.path.join(ROOT, 'Fr5_intertek_2nd_250526/matched_index_with_roi.csv'),
                        os.path.join(ROOT, 'Fr5_intertek_3rd_250526/matched_index_with_roi.csv'),
                        os.path.join(ROOT, 'Fr5_intertek_4th_250526/matched_index_with_roi.csv'),
                        os.path.join(ROOT, 'Fr5_intertek_5th_250526/matched_index_with_roi.csv'),
                        os.path.join(ROOT, 'Fr5_intertek_6th_250526/matched_index_with_roi.csv'),
                        ]
    val_index_files = [os.path.join(ROOT, 'Fr5_intertek_7th_250526/matched_index_with_roi.csv')]
    
    IMG_SIZE = 512
    train_dataset = RobotArmSegFKDataset(index_paths=train_index_files, image_size=IMG_SIZE)
    val_dataset = RobotArmSegFKDataset(index_paths=val_index_files, image_size=IMG_SIZE)
    
    model = PoseEstimationHRViT(num_kp=7, pretrained=True, img_size=IMG_SIZE).to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    ## --- W&B 연동 3: wandb.watch로 모델 그래디언트 추적 (메인 프로세스에서만) ---
    if local_rank == 0:
        wandb.watch(model, log="all", log_freq=100) # 100 스텝마다 그래디언트 기록

    train_and_validate(
        local_rank=local_rank,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
    )


if __name__ == "__main__":
    # # 데이터셋 인덱싱 (한 번만 실행하면 됩니다)
    # subdirs = [
    #     os.path.join(ROOT, d) for d in os.listdir(ROOT)
    #     if os.path.isdir(os.path.join(ROOT, d)) and d.startswith("Fr5_intertek_")
    # ]
    # subdirs.sort()
    # print(f"총 {len(subdirs)}개 데이터셋 세트 탐색: {subdirs}")
    # for sd in subdirs:
    #     process_dataset_indexing(sd, MAX_TIME_DIFF)

    print("메인 학습 스크립트를 시작합니다. (인덱싱은 생략)")
    main()