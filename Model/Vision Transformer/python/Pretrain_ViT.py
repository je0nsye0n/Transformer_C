import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# weights 및 transform 구성
weights = ViT_B_16_Weights.IMAGENET1K_V1
transform = weights.transforms()  # 필요한 전처리 파이프라인 포함

# 사전학습된 모델 로드 (분류 헤드 포함: 1000 클래스)
model = vit_b_16(weights=weights)
model.eval()

# ImageNet 클래스 이름 목록 가져오기
imagenet_labels = weights.meta["categories"]

# 데이터 경로 지정
data_dir = './Data'
image_extensions = ('.jpeg', '.jpg', '.png')
image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(image_extensions)]

# 결과 로그 파일 열기 (새 파일로 저장하고 싶으면 'w' 모드, 이어쓰기라면 'a' 모드)
log_path = './Resultlog.txt'
with open(log_path, 'w', encoding='utf-8') as log_file:
    if not image_files:
        message = "데이터 폴더에 이미지가 없습니다."
        print(message)
        log_file.write(message + "\n")
    else:
        for img_path in image_files:
            try:
                # 이미지 로드 및 전처리
                img = Image.open(img_path).convert('RGB')
                x = transform(img).unsqueeze(0)  # 배치 차원 추가

                # 추론 수행
                with torch.no_grad():
                    logits = model(x)
                    probabilities = F.softmax(logits, dim=-1)
                    pred_idx = probabilities.argmax(dim=-1).item()
                
                # 인덱스를 클래스 이름으로 변환
                pred_label = imagenet_labels[pred_idx]
                result_str = f"이미지: {os.path.basename(img_path)} -> 예측 클래스: {pred_label} (인덱스: {pred_idx})"
                
                # 콘솔 출력 및 로그 파일에 기록
                #print(result_str)
                log_file.write(result_str + "\n")

            except Exception as e:
                error_msg = f"{os.path.basename(img_path)} 처리 중 오류 발생: {e}"
                print(error_msg)
                log_file.write(error_msg + "\n")
