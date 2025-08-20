import os
import cv2
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from my_models import SegFormerForRobotArm
from my_utils import mask_to_bbox, crop_with_padding

# ================= 설정 =================
ROOT = "/home/najo/NAS/DIP/datasets/Fr5_intertek_dataset"
SEGFORMER_MODEL_PATH = "/home/najo/NAS/DIP/Fr5_robot_SegFormer/best_segformer_robot_arm.pth"
ROI_OUTPUT_DIR = os.path.join(ROOT, "preprocessed_roi")
# ========================================

# --- 각 워커 프로세스를 위한 전역 변수 ---
g_model = None
g_device = None
g_transform = None

# --- 수정 1: 워커 초기화 함수를 더 간단하게 변경 ---
def init_worker(model_path):
    """
    각 워커 프로세스가 시작될 때 호출됩니다.
    워커 자신의 ID를 기반으로 GPU를 할당합니다.
    """
    global g_model, g_device, g_transform
    
    # 자신의 프로세스 ID를 확인하여 GPU 번호를 결정
    worker_id = mp.current_process()._identity[0] - 1
    num_gpus = torch.cuda.device_count()
    device_id = worker_id % num_gpus
    g_device = f"cuda:{device_id}"
    torch.cuda.set_device(g_device)

    # 모델을 현재 프로세스의 GPU에 로드
    g_model = SegFormerForRobotArm(num_classes=2, model_name="nvidia/mit-b2").to(g_device)
    checkpoint = torch.load(model_path, map_location=g_device, weights_only=True)
    g_model.load_state_dict(checkpoint['model_state_dict'])
    g_model.eval()

    g_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def process_row(row_dict):
    """단일 이미지(row)를 처리하는 함수"""
    global g_model, g_device, g_transform
    
    # ... (내부 로직은 이전과 동일) ...
    dir_name = row_dict['dir_name']
    img_path = row_dict["img.path"]

    if not os.path.exists(img_path): return None
        
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = rgb.shape[:2]
    
    with torch.no_grad():
        transformed = g_transform(image=rgb) 
        inp = transformed['image'].unsqueeze(0).to(g_device)
        logits = g_model(inp)
        fg_prob = torch.softmax(logits, dim=1)[0][1]
        mask_512 = F.interpolate(fg_prob.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
        mask_512 = (mask_512.squeeze() > 0.5).cpu().numpy().astype(np.uint8)

    box_512 = mask_to_bbox(mask_512, min_area=100)
    
    new_row = row_dict
    if box_512 is not None:
        x1_512, y1_512, x2_512, y2_512 = box_512
        x_scale, y_scale = original_w / 512, original_h / 512
        x1, y1 = int(x1_512 * x_scale), int(y1_512 * y_scale)
        x2, y2 = int(x2_512 * x_scale), int(y2_512 * y_scale)
        original_box = (x1, y1, x2, y2)
        crop = crop_with_padding(rgb, original_box, pad=10)
        base_filename = os.path.basename(img_path)
        roi_filename = f"{dir_name}_{base_filename}"
        roi_save_path = os.path.join(ROI_OUTPUT_DIR, roi_filename)
        cv2.imwrite(roi_save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        new_row['roi.path'] = roi_save_path
        new_row['roi.x1'], new_row['roi.y1'], new_row['roi.x2'], new_row['roi.y2'] = original_box
    else:
        new_row['roi.path'] = None
        new_row['roi.x1'], new_row['roi.y1'], new_row['roi.x2'], new_row['roi.y2'] = [None] * 4
        
    return new_row

def main():
    os.makedirs(ROI_OUTPUT_DIR, exist_ok=True)
    
    all_records_to_process = []
    subdirs = [d for d in os.listdir(ROOT) if d.startswith("Fr5_intertek_")]
    subdirs.sort()

    for dir_name in subdirs:
        index_file = os.path.join(ROOT, dir_name, "matched_index.csv")
        if not os.path.exists(index_file): continue
        df = pd.read_csv(index_file)
        df['dir_name'] = dir_name
        all_records_to_process.extend(df.to_dict('records'))

    if not all_records_to_process:
        print("No data to process.")
        return

    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        pass
    
    num_gpus = torch.cuda.device_count()
    num_processes = num_gpus if num_gpus > 0 else 1

    # --- 수정 2: Pool 생성 코드를 더 간단하고 명확하게 변경 ---
    # initargs는 모든 워커에게 동일한 인자를 전달합니다.
    with mp.Pool(processes=num_processes, initializer=init_worker, initargs=(SEGFORMER_MODEL_PATH,)) as pool:
        updated_records = []
        with tqdm(total=len(all_records_to_process)) as pbar:
            for result in pool.imap_unordered(process_row, all_records_to_process):
                if result is not None:
                    updated_records.append(result)
                pbar.update()

    if updated_records:
        print("\nProcessing finished. Aggregating results and saving to CSVs.")
        final_df = pd.DataFrame(updated_records)
        for dir_name, group_df in final_df.groupby('dir_name'):
            group_df = group_df.drop(columns=['dir_name'])
            output_csv_path = os.path.join(ROOT, dir_name, "matched_index_with_roi.csv")
            group_df.to_csv(output_csv_path, index=False)
            print(f"Successfully created: {output_csv_path}")

if __name__ == "__main__":
    main()