import os
import cv2
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

# 기존에 만드신 모듈들을 가져옵니다.
import albumentations as A
from albumentations.pytorch import ToTensorV2
from my_models import SegFormerForRobotArm
from my_utils import mask_to_bbox, crop_with_padding

# ================= 설정 =================
ROOT = "/home/najo/NAS/DIP/datasets/Fr5_intertek_dataset"
SEGFORMER_MODEL_PATH = "/home/najo/NAS/DIP/Fr5_robot_SegFormer/best_segformer_robot_arm.pth"
ROI_OUTPUT_DIR = os.path.join(ROOT, "preprocessed_roi") # ROI 이미지를 저장할 폴더
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 단일 GPU 사용
# ========================================

def main():
    """
    모든 데이터셋의 이미지에 대해 세그멘테이션을 실행하여 ROI를 추출하고,
    결과 이미지와 바운딩 박스 정보가 포함된 새로운 CSV 파일을 생성합니다.
    """
    print(f"Using device for preprocessing: {DEVICE}")
    os.makedirs(ROI_OUTPUT_DIR, exist_ok=True)

    # 1. SegFormer 모델 로드 (GPU 0번에만 올립니다)
    print("Loading SegFormer model...")
    seg_model = SegFormerForRobotArm(num_classes=2, model_name="nvidia/mit-b2").to(DEVICE)
    checkpoint = torch.load(SEGFORMER_MODEL_PATH, map_location=DEVICE)
    seg_model.load_state_dict(checkpoint['model_state_dict'])
    seg_model.eval()
    print("Model loaded successfully.")
    
    seg_input_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

    # 2. 모든 데이터셋 폴더를 순회
    subdirs = [d for d in os.listdir(ROOT) if d.startswith("Fr5_intertek_")]
    subdirs.sort()

    for dir_name in subdirs:
        dataset_path = os.path.join(ROOT, dir_name)
        index_file = os.path.join(dataset_path, "matched_index.csv")
        output_csv_path = os.path.join(dataset_path, "matched_index_with_roi.csv")

        if not os.path.exists(index_file):
            print(f"Skipping {dir_name}: matched_index.csv not found.")
            continue

        print(f"\nProcessing dataset: {dir_name}")
        df = pd.read_csv(index_file)
        
        # 결과를 저장할 리스트
        updated_records = []
        
        # 3. 각 이미지에 대해 ROI 추출 및 저장
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ROIs"):
            img_path = row["img.path"]
            if not os.path.exists(img_path):
                continue
                
            bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # PIL.Image.fromarray(rgb) 이 라인은 이제 필요 없습니다.

            # 세그멘테이션 추론
            with torch.no_grad():
                # --- 이 부분을 아래와 같이 수정 ---
                # 1. NumPy 배열인 rgb를 'image'라는 이름으로 전달
                transformed = seg_input_transform(image=rgb) 
                # 2. 결과 딕셔너리에서 'image' 키로 텐서를 가져옴
                inp = transformed['image'].unsqueeze(0).to(DEVICE)
                # ---------------------------------
                
                logits = seg_model(inp)
                probs = torch.softmax(logits, dim=1)[0]
                fg_prob = probs[1] # fg_class_id=1
                mask = (fg_prob > 0.5).float().cpu().numpy().astype(np.uint8)

            box = mask_to_bbox(mask, min_area=100)
            
            new_row = row.to_dict()
            if box is not None:
                # ROI 이미지 크롭 및 저장
                crop = crop_with_padding(rgb, box, pad=10)
                
                # 저장할 파일명 생성 (원본 파일명 기반)
                base_filename = os.path.basename(img_path)
                roi_filename = f"{dir_name}_{base_filename}"
                roi_save_path = os.path.join(ROI_OUTPUT_DIR, roi_filename)
                
                # BGR로 변환하여 저장
                cv2.imwrite(roi_save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

                # 새 CSV에 저장할 정보 추가
                new_row['roi.path'] = roi_save_path
                new_row['roi.x1'], new_row['roi.y1'], new_row['roi.x2'], new_row['roi.y2'] = box
            else:
                # 바운딩 박스를 찾지 못한 경우
                new_row['roi.path'] = None
                new_row['roi.x1'], new_row['roi.y1'], new_row['roi.x2'], new_row['roi.y2'] = [None] * 4

            updated_records.append(new_row)

        # 4. 새로운 정보가 담긴 CSV 파일 저장
        if updated_records:
            new_df = pd.DataFrame(updated_records)
            new_df.to_csv(output_csv_path, index=False)
            print(f"Successfully created: {output_csv_path}")

if __name__ == "__main__":
    main()