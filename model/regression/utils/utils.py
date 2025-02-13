import os
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm
from sklearn.metrics import r2_score  # R² Score 계산
import rasterio
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # device 설정

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10, task_type="classification"):
    task_type = task_type.lower()
    if task_type in ["classification", "c"]:
        task_type = "classification"
    elif task_type in ["regression", "r"]:
        task_type = "regression"
    else:
        raise ValueError("Invalid task_type. Use 'classification' (or 'c') or 'regression' (or 'r').")
        
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model.state_dict())
    no_improve_count = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if task_type == "regression":
                # Regression일 경우 flatten하여 (batch, ?) 형태로 변환
                outputs = outputs.view(outputs.shape[0], -1)
                labels = labels.view(labels.shape[0], -1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            if task_type == "classification":
                _, predicted = torch.max(outputs, 1)
            else:  # Regression
                predicted = outputs  # 그대로 사용
            
            all_preds.extend(predicted.cpu().detach().numpy().flatten())  # flatten 추가
            all_labels.extend(labels.cpu().detach().numpy().flatten())   # flatten 추가

        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        if task_type == "classification":
            train_metric = 100 * (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
        else:  # Regression
            train_metric = r2_score(all_labels, all_preds)  # R² Score

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                if task_type == "regression":
                    outputs = outputs.view(outputs.shape[0], -1)
                    labels = labels.view(labels.shape[0], -1)

                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * labels.size(0)

                if task_type == "classification":
                    _, predicted = torch.max(outputs, 1)
                else:  # Regression
                    predicted = outputs  # 그대로 사용

                val_preds.extend(predicted.cpu().detach().numpy().flatten())  # flatten 추가
                val_labels.extend(labels.cpu().detach().numpy().flatten())   # flatten 추가

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if task_type == "classification":
            val_metric = 100 * (torch.tensor(val_preds) == torch.tensor(val_labels)).sum().item() / len(val_labels)
            metric_name = "Accuracy"
        else:  # Regression
            val_metric = r2_score(val_labels, val_preds)
            metric_name = "R² Score"

        print(f"\nEpoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train {metric_name}: {train_metric:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val {metric_name}: {val_metric:.4f}\n")

        # Early Stopping & Model Saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    return best_model_state, train_losses, val_losses

class TiffDataset(Dataset):
    def __init__(self, large_tif_dir, label_tif_dir, file_list, patch_size=3, stride=1, box_filter_fn=None, transform=None):
        """
        Args:
            large_tif_dir (str): 원천 데이터가 저장된 디렉토리
            label_tif_dir (str): 레이블 TIFF가 저장된 디렉토리
            file_list (list of str): 처리할 파일명 리스트
            patch_size (int): 슬라이싱할 이미지의 크기 (항상 홀수)
            stride (int): 패치 슬라이싱 간격
            box_filter_fn (callable, optional): 특정 박스를 필터링하는 함수
            transform (callable, optional): 데이터 변환 함수
        """
        if patch_size % 2 == 0:
            raise ValueError("patch_size는 홀수여야 합니다.")
        if stride < 1:
            raise ValueError("stride는 1 이상이어야 합니다.")
        
        self.large_tif_dir = large_tif_dir
        self.label_tif_dir = label_tif_dir
        self.file_list = set(file_list)
        self.patch_size = patch_size
        self.stride = stride
        self.half_size = patch_size // 2
        self.box_filter_fn = box_filter_fn or (lambda box_number: True)
        self.transform = transform
        
        self.image_cache = {}
        self.label_cache = {}
        self.sample_list = self._generate_sample_list()
    
    def _load_image(self, file_name, is_label=False):
        """TIFF 이미지를 로드하고 캐싱"""
        cache = self.label_cache if is_label else self.image_cache
        directory = self.label_tif_dir if is_label else self.large_tif_dir
        
        if file_name not in cache:
            file_path = os.path.join(directory, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file '{file_path}' not found.")
            image = tiff.imread(file_path)
            
            # 레이블 데이터: (H, W) → (H, W, 1) 변환
            if is_label:
                image = image[:, :, np.newaxis]
            
            cache[file_name] = image
        return cache[file_name]
    
    def _generate_sample_list(self):
        """(filename, center coordinate, box) 리스트 생성"""
        sample_list = []
        for file_name in self.file_list:
            try:
                image = self._load_image(file_name)
                h, w = image.shape[:2]
                
                # stride를 반영한 좌표 생성
                for y in range(self.half_size, h - self.half_size, self.stride):
                    for x in range(self.half_size, w - self.half_size, self.stride):
                        box_number = (y // 300) * 10 + (x // 300) + 1
                        if self.box_filter_fn(box_number):
                            sample_list.append((file_name, (x, y)))
            except FileNotFoundError:
                continue  # 파일이 없으면 스킵
        return sample_list
    
    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, idx):
        file_name, (x, y) = self.sample_list[idx]
        image = self._load_image(file_name)
        label = self._load_image(file_name, is_label=True)
        
        # 패치 슬라이싱
        x_start, x_end = x - self.half_size, x + self.half_size + 1
        y_start, y_end = y - self.half_size, y + self.half_size + 1
        patch = image[y_start:y_end, x_start:x_end, :]
        label_patch = label[y_start:y_end, x_start:x_end, :]
        
        if self.transform:
            patch, label_patch = self.transform(patch, label_patch)
        
        return patch, label_patch

#patch 와 label에 각각 다른 transform을 적용할 수 있도록 한다.
class DualTransform:
    def __init__(self, source_transform, label_transform):
        self.source_transform = source_transform
        self.label_transform = label_transform

    def __call__(self, patch, label_patch):
        patch = self.source_transform(patch)
        label_patch = self.label_transform(label_patch)  # 레이블도 동일하게 변환 적용
        return patch, label_patch

# 큰 이미지에 모델 적용하는 함수 patch_szie < stride이면 중앙의 stride*stride크기의 부분이 해당하는 값으로 mapping된다.
def process_large_image(model, image_path, patch_size=5, stride=3):
    device = next(model.parameters()).device  # 모델의 디바이스 확인
    
    with rasterio.open(image_path) as src:
        image = src.read()  # (108, 3000, 3000)
    
    c, h, w = image.shape
    pad = patch_size // 2
    padded_image = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), mode='reflect')
    
    output = np.zeros((h, w), dtype=np.float32)
    
    def process_patch(i, j):
        patch = padded_image[:, i:i+patch_size, j:j+patch_size]
        patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)  # GPU로 이동
        with torch.no_grad():
            pred = model(patch_tensor).cpu().item()  # CPU로 다시 이동 후 숫자로 변환
        return i, j, pred
    
    indices = [(i, j) for i in range(0, h, stride) for j in range(0, w, stride)]
    
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda x: process_patch(*x), indices), total=len(indices)))
    
    for i, j, pred in results:
        output[i:i+stride, j:j+stride] = pred  # Assign predicted value to stride region
    
    return output

# 결과 시각화 함수
def visualize_result(result, title = 'Visualization'):
    result = result.squeeze()  # Ensure it's 2D
    plt.figure(figsize=(10, 10))
    plt.imshow(result, cmap='Greens')  # Use green colormap for grayscale
    plt.colorbar()
    plt.title(title)
    plt.show()

#he initializer 함수
def he_init_weights(m):
    """
    Applies He (Kaiming) initialization to Conv layers and Linear layers.
    
    - Conv2d, Conv3d: He Normal Initialization with fan_out mode
    - Linear: He Uniform Initialization with fan_in mode
    - BatchNorm: gamma = 1, beta = 0
    - Biases (if exist): Initialized to zero
    """
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)
    
    elif isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)