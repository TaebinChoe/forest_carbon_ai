import os
import torch
from torchvision import transforms
from utils.utils import train_model, TiffDataset, DualTransform, process_large_image, visualize_result, he_init_weights
from models.cnn import CNNRegressor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import rasterio
import gc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = 5
    stride = 5
    num_epochs = 50
    
    file_list = ["jiri_1.tif", "jiri_2.tif", "sobaek.tif"]  # 사용할 TIFF 파일 리스트
    checkpoints_dir = "./checkpoints/use_patch" #check point 저장할 폴더
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    source_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.float()),  # uint16 → float 변환 -> squeeze
    ])

    label_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float().mean(dim=[1, 2], keepdim=False))  # 평균 계산 후 (1,) 유지
    ])

    # DualTransform을 적용하여 입력과 레이블을 변환
    dual_transform = DualTransform(source_transform, label_transform)
    
    test_filter = lambda box_number: (box_number % 9 == 0 or box_number % 9 == 5) #test지역 선별
    
    #for source_dir, in_channel, add in zip(['with_s2', 'with_s2_spc'], [108, 114], ['', '_spc']):
    for source_dir, in_channel, add in zip(['with_s2_spc'], [114], ['_spc']):
        large_tif_dir = f"../../data/source_data/{source_dir}/"  # 원천 데이터 TIFF 파일이 있는 폴더
        for target in ["height", "density", "DBH"]:
            label_dir = f"../../data/label_data/{target}"  # 대응하는 레이블 TIFF 파일이 있는 폴더

            # 데이터셋 생성
            train_dataset = TiffDataset(large_tif_dir, label_dir, file_list, patch_size=patch_size, stride=stride, box_filter_fn=lambda box_number: not test_filter(box_number), transform=dual_transform)
            val_dataset = TiffDataset(large_tif_dir, label_dir, file_list, patch_size=patch_size, stride=stride, box_filter_fn=test_filter, transform=dual_transform)

            # DataLoader 설정
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
            
            model = CNNRegressor(in_channel).to(device)
            model.apply(he_init_weights)

            # 손실 함수
            criterion = nn.MSELoss()  # 평균제곱오차 손실

            # 옵티마이저 (Adam 추천)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            best_model_state, train_losses, val_losses = train_model(
                model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, patience=100, task_type = "r"
            )
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, os.path.join(checkpoints_dir, f"cnn_{target}{add}_{patch_size}_{num_epochs}.pth"))
            
if __name__ == "__main__":
    main()