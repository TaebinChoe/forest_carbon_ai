import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import tifffile as tiff
from torch.utils.data import Dataset
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import copy
import torch.nn as nn
import torch.nn.init as init
import random
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device 설정

target_name_mapping = {
    0: "NF", #Non-Forest, 비산림
    1: "PD", # Pinus densiflora, 소나무
    2: "PK", # Pinus koraiensis, 잣나무
    3: "LK", # Larix kaempferi, 낙엽송
    4: "QM", # Quercus mongolica, 신갈나무
    5: "QV" # Quercus variabilis, 굴참나무
}

# 날짜 라벨 설정
dates = ['0201', '0301', '0401', '0415', '0501', '0515', '0601', '0701', '0901', '1001', '1015', '1101']

"""
모델 학습 함수
"""
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None, patience = None):
    
    if patience == None:
        patience= num_epochs
        
    train_losses = []
    val_losses = []
    best_val_f1_score = 0
    best_model_state = model.state_dict()  # 초기 모델 가중치 저장
    no_improve_count = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_running_loss = 0.0
        
        train_labels = []
        train_predictions = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * labels.size(0)  # 배치별 loss * 개수로 전체 손실 계산
            _, predicted = torch.max(outputs, 1)
            
            train_labels.extend(labels.cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())
            
        train_loss = train_running_loss / len(train_loader.dataset)  # 전체 샘플 수로 나눔
        train_losses.append(train_loss)
        
        full_class_labels = np.arange(outputs.shape[1])
        train_report = classification_report(train_labels, train_predictions, labels=full_class_labels, output_dict=True)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0

        val_labels = []
        val_predictions = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                
                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_report = classification_report(val_labels, val_predictions, labels=full_class_labels, output_dict=True)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_report['accuracy']:.2f}, Train f1-score: {train_report["macro avg"]["f1-score"]:.2f} "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_report['accuracy']:.2f}, Val f1-score: {val_report["macro avg"]["f1-score"]:.2f}\n")
        
        # Early Stopping & Model Saving
        if best_val_f1_score < val_report["macro avg"]["f1-score"]:
            best_val_f1_score = val_report["macro avg"]["f1-score"]
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print("Early stopping triggered. Training stopped.")
            break
            
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # validation loss 기준
            else:
                scheduler.step()  # 일반적인 step()
    
    return best_model_state, train_losses, val_losses

"""
모델 평가를 위한 함수들
"""
def evaluate_model_with_cm(model, val_loader, num_classes=6, target_name_mapping=target_name_mapping):
    """
    모델의 성능을 평가하고 혼동 행렬(Confusion Matrix) 및 분류 리포트를 생성하는 함수.
    추가적으로 침엽수와 활엽수의 분류력 및 내부 분류력을 분석하고 이를 반환하는 데이터프레임에 포함한다.
    """
    model.eval()
    all_labels = []
    all_predictions = []

    # 검증 데이터셋을 이용하여 모델 예측 수행
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluation Progress"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 전체 클래스 라벨 리스트 생성
    full_class_labels = np.arange(num_classes)
    cm = confusion_matrix(all_labels, all_predictions, labels=full_class_labels)
    
    # 전체 데이터에 대한 혼동 행렬 시각화
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_name_mapping.values(), yticklabels=target_name_mapping.values())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 전체 데이터에 대한 분류 리포트
    target_names = list(target_name_mapping.values())
    report_dict = classification_report(all_labels, all_predictions, labels=full_class_labels, target_names=target_names, digits=3, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df["task"] = "Overall"
    
    # 추가적인 분석 수행 및 결과 저장
    conifer_vs_broadleaf_report = evaluate_conifer_vs_broadleaf(all_labels, all_predictions) #침/활 분류력
    conifer_vs_broadleaf_report["task"] = "Conifer vs Broadleaf"
    
    conifer_report = evaluate_conifer_classification(all_labels, all_predictions) #침엽수 내 분류력
    conifer_report["task"] = "Conifer"
    
    broadleaf_report = evaluate_broadleaf_classification(all_labels, all_predictions) #활엽수 내 분류력
    broadleaf_report["task"] = "Broadleaf"
    
    # 결과를 하나의 데이터프레임으로 통합
    additional_metrics = pd.concat([conifer_vs_broadleaf_report, conifer_report, broadleaf_report])
    final_report_df = pd.concat([report_df, additional_metrics])
    
    return final_report_df

# 침엽수 vs 활엽수 분류력 평가 함수
def evaluate_conifer_vs_broadleaf(all_labels, all_predictions):
    """
    침엽수와 활엽수를 구분하는 능력을 평가하는 함수.
    Non-Forest를 제외한 데이터에서 침엽수와 활엽수를 얼마나 잘 분류했는지를 평가한다.
    """
    conifer_classes = {1, 2, 3}  # 침엽수 클래스
    broadleaf_classes = {4, 5}  # 활엽수 클래스
    
    # Non-Forest 제외 후 침엽수와 활엽수로만 그룹핑
    valid_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_predictions)) if (label in conifer_classes or label in broadleaf_classes) and (pred in conifer_classes or pred in broadleaf_classes)]
    true_labels = [1 if all_labels[i] in conifer_classes else 2 for i in valid_indices]
    pred_labels = [1 if all_predictions[i] in conifer_classes else 2 for i in valid_indices]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=[1, 2])
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Conifer", "Broadleaf"], yticklabels=["Conifer", "Broadleaf"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Conifer vs Broadleaf Classification')
    plt.show()
    
    return pd.DataFrame(classification_report(true_labels, pred_labels, target_names=["Conifer", "Broadleaf"], digits=3, output_dict=True)).transpose()

# 침엽수 내부 분류력 평가 함수
def evaluate_conifer_classification(all_labels, all_predictions):
    """
    침엽수 내부에서 개별 클래스를 구분하는 능력을 평가하는 함수.
    """
    conifer_classes = {1, 2, 3}
    valid_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_predictions)) if label in conifer_classes and pred in conifer_classes]
    true_labels = [all_labels[i] for i in valid_indices]
    pred_labels = [all_predictions[i] for i in valid_indices]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=[1, 2, 3])
    target_names = [target_name_mapping[1], target_name_mapping[2], target_name_mapping[3]]
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Conifer Classification')
    plt.show()
    
    return pd.DataFrame(classification_report(true_labels, pred_labels, target_names=target_names, digits=3, output_dict=True)).transpose()

# 활엽수 내부 분류력 평가 함수
def evaluate_broadleaf_classification(all_labels, all_predictions):
    """
    활엽수 내부에서 개별 클래스를 구분하는 능력을 평가하는 함수.
    """
    broadleaf_classes = {4, 5}
    valid_indices = [i for i, (label, pred) in enumerate(zip(all_labels, all_predictions)) if label in broadleaf_classes and pred in broadleaf_classes]
    true_labels = [all_labels[i] for i in valid_indices]
    pred_labels = [all_predictions[i] for i in valid_indices]
    
    cm = confusion_matrix(true_labels, pred_labels, labels=[4, 5])
    target_names = [target_name_mapping[4], target_name_mapping[5]]
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Broadleaf Classification')
    plt.show()
    
    return pd.DataFrame(classification_report(true_labels, pred_labels, target_names=target_names, digits=3, output_dict=True)).transpose()

#rf평가를 위한 함수
def evaluate_rf_model_with_cm(model, val_loader, num_classes=6, target_name_mapping=target_name_mapping):
    """
    Random Forest 모델의 성능을 평가하고 혼동 행렬(Confusion Matrix) 및 분류 리포트를 생성하는 함수.
    추가적으로 침엽수와 활엽수의 분류력 및 내부 분류력을 분석하고 이를 반환하는 데이터프레임에 포함한다.
    """
    all_labels = []
    all_predictions = []

    # 검증 데이터셋을 이용하여 모델 예측 수행
    for X_batch, y_batch in tqdm(val_loader, desc="Evaluation Progress"):
        X_batch = X_batch.numpy()  # NumPy로 변환 (RandomForest는 NumPy 배열을 사용)
        y_batch = y_batch.numpy()
        
        # 예측
        preds = model.predict(X_batch)  # RandomForest 예측

        all_labels.extend(y_batch)
        all_predictions.extend(preds)

    # 전체 클래스 라벨 리스트 생성
    full_class_labels = np.arange(num_classes)
    cm = confusion_matrix(all_labels, all_predictions, labels=full_class_labels)
    
    # 전체 데이터에 대한 혼동 행렬 시각화
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_name_mapping.values(), yticklabels=target_name_mapping.values())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 전체 데이터에 대한 분류 리포트
    target_names = list(target_name_mapping.values())
    report_dict = classification_report(all_labels, all_predictions, labels=full_class_labels, target_names=target_names, digits=3, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df["task"] = "Overall"
    
    # 추가적인 분석 수행 및 결과 저장
    conifer_vs_broadleaf_report = evaluate_conifer_vs_broadleaf(all_labels, all_predictions)  # 침/활 분류력
    conifer_vs_broadleaf_report["task"] = "Conifer vs Broadleaf"
    
    conifer_report = evaluate_conifer_classification(all_labels, all_predictions)  # 침엽수 내 분류력
    conifer_report["task"] = "Conifer"
    
    broadleaf_report = evaluate_broadleaf_classification(all_labels, all_predictions)  # 활엽수 내 분류력
    broadleaf_report["task"] = "Broadleaf"
    
    # 결과를 하나의 데이터프레임으로 통합
    additional_metrics = pd.concat([conifer_vs_broadleaf_report, conifer_report, broadleaf_report])
    final_report_df = pd.concat([report_df, additional_metrics])
    
    return final_report_df


"""
Dataset 클래스
"""
class TiffDataset(Dataset):
    def __init__(self, large_tif_dir, file_list, label_file, patch_size=3, box_filter_fn=None, transform=None):
        """
        Args:
            large_tif_dir (str): 여러 개의 큰 TIFF 파일이 저장된 디렉토리 경로.
            file_list (list of str): 처리할 원천 데이터 파일명 리스트.
            label_file (str): 라벨 정보를 담은 CSV 파일 경로 ("file", "x_pos", "y_pos", "label" 열 포함).
            patch_size (int): 슬라이싱할 이미지의 크기 (항상 홀수여야 함).
            box_filter_fn (callable, optional): 박스 번호 필터링 함수.
            transform (callable, optional): 이미지 전처리에 사용할 함수.
        """
        if patch_size % 2 == 0:
            raise ValueError("patch_size는 홀수여야 합니다.")

        self.large_tif_dir = large_tif_dir
        self.file_list = set(file_list)
        self.label_df = pd.read_csv(label_file)

        self.box_filter_fn = box_filter_fn or (lambda box_number: True)
        self.label_df = self.label_df[self.label_df['file'].isin(self.file_list)].reset_index(drop=True)
        self.label_df = self.label_df[self.label_df['box_number'].apply(self.box_filter_fn)].reset_index(drop=True)

        self.patch_size = patch_size
        self.half_size = patch_size // 2
        self.transform = transform

        self.image_cache = {}

        # 유효한 샘플만 남긴 인덱스 리스트
        self.valid_indices = self._filter_valid_indices()

    def _load_image(self, file_name):
        if file_name not in self.image_cache:
            file_path = os.path.join(self.large_tif_dir, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file '{file_path}' not found.")
            self.image_cache[file_name] = tiff.imread(file_path)
        return self.image_cache[file_name]

    def _filter_valid_indices(self):
        """이미지 경계를 벗어나지 않는 유효한 인덱스만 저장"""
        valid_indices = []
        for idx, row in self.label_df.iterrows():
            file_name = row["file"]
            x, y = row["x_pos"], row["y_pos"]

            # 이미지 크기 확인
            try:
                image = self._load_image(file_name)
                image_height, image_width = image.shape[:2]

                x_start, y_start = x - self.half_size, y - self.half_size
                x_end, y_end = x + self.half_size + 1, y + self.half_size + 1

                # 이미지 범위를 벗어나지 않는 경우만 유효한 인덱스로 저장
                if 0 <= x_start and 0 <= y_start and x_end <= image_width and y_end <= image_height:
                    valid_indices.append(idx)
            except FileNotFoundError:
                continue  # 파일이 없으면 스킵

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.label_df.iloc[self.valid_indices[idx]]
        file_name, x, y, label = row["file"], row["x_pos"], row["y_pos"], row["label"]

        image = self._load_image(file_name)
        x_start, y_start = x - self.half_size, y - self.half_size
        x_end, y_end = x + self.half_size + 1, y + self.half_size + 1

        patch = image[y_start:y_end, x_start:x_end]

        if self.transform:
            patch = self.transform(patch)

        return patch, label

"""
기본 데이터 transform
"""
class ReshapeTransform:
    """(12*bands, 3, 3) → (12, bands, 3, 3) 변환"""
    def __init__(self, bands, patch_size, time=12):
        self.bands = bands
        self.patch_size = patch_size
        self.time = time

    def __call__(self, x):
        #return x.view(self.bands, self.time, self.patch_size, self.patch_size)
        return x.view(self.time, self.bands, self.patch_size, self.patch_size).permute(1,0,2,3)

def scale_up_planet_channels(x):
    x[:3] *= 5  # 첫 3개 채널을 5배 스케일링
    return x

def base_transform(bands, patch_size, scale_channels=True):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.float()),  # uint16 → float 변환
        ReshapeTransform(bands, patch_size),  # (bands*time, height, width) → (bands, time, height, width)
        transforms.Lambda(scale_up_planet_channels)  # 첫 3개 채널을 5배 확대
    ])
    
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

"""
PSA 함수
"""
import torch
import random
from tqdm import tqdm

def psa_dim(model, data_loader, num_repeats=1, perturbation_strength=0.8, target_dims=None, normalize=True, TP_only=True):
    model.eval()

    sample_batch, _ = next(iter(data_loader))
    num_dims = len(sample_batch.shape) - 1
    dim_names = [f"dim_{i}" for i in range(1, num_dims + 1)]

    # 조사할 차원을 설정 (기본: 모든 차원)
    if target_dims is None:
        target_dims = dim_names
    else:
        target_dims = [f"dim_{i}" for i in target_dims]

    overall_scores = {dim: 0.0 for dim in target_dims}
    per_class_scores = {}
    TP_count = {}  # TP 개수를 저장할 딕셔너리
    total_samples = 0

    for X_batch, targets in tqdm(data_loader, desc="Processing batches"):
        batch_size = X_batch.shape[0]
        total_samples += batch_size
        X_batch = X_batch.to(next(model.parameters()).device)
        targets = targets.to(X_batch.device)

        logit_original = model(X_batch).detach()
        predictions = logit_original.argmax(dim=1)
        num_classes = logit_original.shape[1]

        if not per_class_scores:
            per_class_scores = {cls: {dim: 0.0 for dim in target_dims} for cls in range(num_classes)}
            TP_count = {cls: 0 for cls in range(num_classes)}

        for dim_name in target_dims:
            dim_idx = dim_names.index(dim_name) + 1  # 1-based index
            total_abs_error = 0.0
            class_abs_error = {cls: 0.0 for cls in range(num_classes)}

            for _ in range(num_repeats):
                X_perturbed = X_batch.clone().detach()
                num_swap = max(1, int(X_perturbed.shape[dim_idx] * perturbation_strength))
                swap_indices = random.sample(range(X_perturbed.shape[dim_idx]), num_swap)
                permutation = random.sample(swap_indices, len(swap_indices))

                X_perturbed.index_copy_(dim_idx, torch.tensor(swap_indices, device=X_batch.device),
                                        X_perturbed.index_select(dim_idx, torch.tensor(permutation, device=X_batch.device)))

                logit_perturbed = model(X_perturbed).detach()
                abs_error = torch.abs(logit_original - logit_perturbed)  # 원본과 변형된 예측값의 절대 차이 계산

                total_abs_error += abs_error.mean().item() * batch_size

                for cls in range(num_classes):
                    if TP_only:
                        TP_mask = (predictions == targets) & (targets == cls)
                        TP_sample_count = TP_mask.sum().item()
                        if TP_sample_count > 0:
                            TP_count[cls] += TP_sample_count
                            class_abs_error[cls] += abs_error[TP_mask].sum().item()  # TP 개별 샘플들의 에러 합산
                    else:
                        class_abs_error[cls] += abs_error[:, cls].sum().item()

            overall_scores[dim_name] += total_abs_error / num_repeats
            for cls in range(num_classes):
                per_class_scores[cls][dim_name] += class_abs_error[cls] / num_repeats

    for key in overall_scores:
        overall_scores[key] /= total_samples

    for cls in per_class_scores:
        if TP_only and TP_count[cls] > 0:
            for key in per_class_scores[cls]:
                per_class_scores[cls][key] /= TP_count[cls]  # TP 샘플 개수로 나누어 평균 계산
        else:
            for key in per_class_scores[cls]:
                per_class_scores[cls][key] /= total_samples

    # 🔹 선택한 차원만 정규화하여 합이 1이 되도록 조정
    if normalize:
        total_score = sum(overall_scores.values())
        overall_scores = {key: value / total_score for key, value in overall_scores.items()} if total_score > 0 else overall_scores

        for cls in per_class_scores:
            class_total_score = sum(per_class_scores[cls].values())
            per_class_scores[cls] = {key: value / class_total_score for key, value in per_class_scores[cls].items()} if class_total_score > 0 else per_class_scores[cls]

    return {"overall": overall_scores, "per_class": per_class_scores}


def plot_psa_dim_scores(overall_scores, per_class_scores, dim_labels=["Bands", "Time", "Space"]):
    # Overall Dimension Importance (Bar Chart)
    combined_scores = {
        "Bands": overall_scores['dim_1'],
        "Time": overall_scores['dim_2'],
        "Space": overall_scores['dim_3'] + overall_scores['dim_4']
    }

    plt.figure(figsize=(8, 5))
    plt.bar(dim_labels, combined_scores.values(), color='skyblue')
    plt.xlabel("Dimension")
    plt.ylabel("Sensitivity Score")
    plt.title("Overall Dimension Sensitivity")
    plt.xticks(rotation=45)
    plt.show()

    # Per-Class Importance (Heatmap)
    per_class_df = {}
    for cls, scores in per_class_scores.items():
        per_class_df[target_name_mapping.get(cls, f"Class {cls}")] = [
            scores['dim_1'],
            scores['dim_2'],
            scores['dim_3'] + scores['dim_4']  # Space
        ]
        
    plt.figure(figsize=(10, 6))
    sns.heatmap(list(per_class_df.values()), annot=True, cmap="Blues", xticklabels=dim_labels, yticklabels=list(per_class_df.keys()))
    plt.xlabel("Dimension")
    plt.ylabel("Class")
    plt.title("Per-Class Dimension Sensitivity")
    plt.xticks(rotation=45)
    plt.show()


#perturbation 관련 코드
def psa_bands_time(model, dataloader, num_classes=6, num_repeats=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    band, time = 4, 12  # 데이터의 밴드 수와 시간 스텝 수

    # 결과 저장 변수
    total_sensitivity_map = torch.zeros((num_classes, band, time), device=device)
    total_count_map = torch.zeros((num_classes, band, time), device=device)

    total_band_sensitivity = torch.zeros((num_classes, band), device=device)
    total_band_count = torch.zeros((num_classes, band), device=device)

    total_time_sensitivity = torch.zeros((num_classes, time), device=device)
    total_time_count = torch.zeros((num_classes, time), device=device)

    with torch.no_grad():
        for repeat in range(num_repeats):
            sensitivity_map = torch.zeros((num_classes, band, time), device=device)
            count_map = torch.zeros((num_classes, band, time), device=device)

            band_sensitivity = torch.zeros((num_classes, band), device=device)
            band_count = torch.zeros((num_classes, band), device=device)

            time_sensitivity = torch.zeros((num_classes, time), device=device)
            time_count = torch.zeros((num_classes, time), device=device)

            for data, labels in tqdm(dataloader, desc=f"Experiment {repeat+1}/{num_repeats}"):
                data, labels = data.to(device), labels.to(device)
                B, Band, Time, H, W = data.shape

                # 1. 원본 예측값 저장
                original_logits = model(data)
                original_preds = original_logits.argmax(dim=1)

                # 2. True Positive(TP) 마스크 생성
                tp_mask = (original_preds == labels)

                # 3. band & time 개별 교란 (Zero-out)
                for b in range(band):
                    for t in range(time):
                        perturbed_data = data.clone()
                        perturbed_data[:, b, t] = 0  # 완전 무효화

                        perturbed_logits = model(perturbed_data)
                        abs_error = torch.abs(original_logits - perturbed_logits)  # 절대값 오차 계산

                        for c in range(num_classes):
                            class_mask = (original_preds == c)
                            valid_mask = tp_mask & class_mask  # TP 중 해당 클래스 데이터 선택

                            sensitivity_map[c, b, t] += abs_error[:, c][valid_mask].sum()
                            count_map[c, b, t] += valid_mask.sum()  # TP 개수 누적

                # 4. band 전체를 Zero-out하여 민감도 측정
                for b in range(band):
                    perturbed_data = data.clone()
                    perturbed_data[:, b] = 0  # 완전 무효화

                    perturbed_logits = model(perturbed_data)
                    abs_error = torch.abs(original_logits - perturbed_logits)

                    for c in range(num_classes):
                        class_mask = (original_preds == c)
                        valid_mask = tp_mask & class_mask

                        band_sensitivity[c, b] += abs_error[:, c][valid_mask].sum()
                        band_count[c, b] += valid_mask.sum()

                # 5. time 전체를 Zero-out하여 민감도 측정
                for t in range(time):
                    perturbed_data = data.clone()
                    perturbed_data[:, :, t] = 0  # 완전 무효화

                    perturbed_logits = model(perturbed_data)
                    abs_error = torch.abs(original_logits - perturbed_logits)

                    for c in range(num_classes):
                        class_mask = (original_preds == c)
                        valid_mask = tp_mask & class_mask

                        time_sensitivity[c, t] += abs_error[:, c][valid_mask].sum()
                        time_count[c, t] += valid_mask.sum()

            # 반복 실험 결과 누적
            total_sensitivity_map += sensitivity_map
            total_count_map += count_map

            total_band_sensitivity += band_sensitivity
            total_band_count += band_count

            total_time_sensitivity += time_sensitivity
            total_time_count += time_count

    # **정규화 (개수로 나누기)**
    total_sensitivity_map = torch.where(total_count_map > 0, total_sensitivity_map / total_count_map, total_sensitivity_map)
    total_band_sensitivity = torch.where(total_band_count > 0, total_band_sensitivity / total_band_count, total_band_sensitivity)
    total_time_sensitivity = torch.where(total_time_count > 0, total_time_sensitivity / total_time_count, total_time_sensitivity)

    return (
        total_sensitivity_map.cpu().numpy(),  # (num_classes, band, time)
        total_band_sensitivity.cpu().numpy(),  # (num_classes, band)
        total_time_sensitivity.cpu().numpy()  # (num_classes, time)
    )

def add_noise(data, noise_level=1.0):
    """
    데이터에 랜덤 노이즈를 추가하는 함수.

    Args:
        data (torch.Tensor): 입력 데이터 (Bands, Time, Height, Width)
        noise_level (float): 노이즈 강도 계수

    Returns:
        torch.Tensor: 노이즈가 추가된 데이터
    """
    noise = (torch.rand_like(data) * 2 - 1) * (data * noise_level)
    return data + noise

def evaluate_perturbation(model, dataloader, num_classes=6, noise_level=0.1, num_repeats=1):
    """
    모델의 교란(노이즈) 영향 평가 및 중요도 히트맵 생성 (기존 + 새로운 방법 포함)

    Args:
        model (torch.nn.Module): 학습된 모델
        dataloader (torch.utils.data.DataLoader): 데이터 로더
        num_classes (int): 클래스 개수
        noise_level (float): 노이즈 강도
        num_repeats (int): 반복 실험 횟수

    Returns:
        tuple:
            - 히트맵 (num_classes, band, time)
            - count_map (num_classes, band, time)
            - band별 중요도 (num_classes, band)
            - time별 중요도 (num_classes, time)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    band, time = 4, 12  # 데이터의 밴드 수와 시간 스텝 수

    # 결과 저장용 변수
    total_heatmap = torch.zeros((num_classes, band, time), device=device)
    total_count_map = torch.zeros((num_classes, band, time), device=device)
    total_band_importance = torch.zeros((num_classes, band), device=device)
    total_time_importance = torch.zeros((num_classes, time), device=device)

    with torch.no_grad():
        for repeat in range(num_repeats):
            heatmap = torch.zeros((num_classes, band, time), device=device)
            band_importance = torch.zeros((num_classes, band), device=device)
            time_importance = torch.zeros((num_classes, time), device=device)

            for data, labels in tqdm(dataloader, desc=f"Experiment {repeat+1}/{num_repeats}"):
                data, labels = data.to(device), labels.to(device)
                B, Band, Time, H, W = data.shape

                # 1. 원본 예측값 저장
                original_logits = model(data)
                original_preds = original_logits.argmax(dim=1)

                # 2. True Positive(TP) 마스크 생성
                tp_mask = (original_preds == labels)

                # 3. 기존 방법: band, time의 특정 값만 교란
                for b in range(band):
                    for t in range(time):
                        perturbed_data = data.clone()
                        perturbed_data[:, b, t] = add_noise(perturbed_data[:, b, t], noise_level)

                        perturbed_logits = model(perturbed_data)
                        mse_loss = F.mse_loss(original_logits, perturbed_logits, reduction='none').mean(dim=1)

                        for c in range(num_classes):
                            class_mask = (original_preds == c)
                            valid_mask = tp_mask & class_mask
                            heatmap[c, b, t] += mse_loss[valid_mask].sum()

                # 4. 새로운 방법 1️⃣: band 전체를 교란하여 중요도 측정
                for b in range(band):
                    perturbed_data = data.clone()
                    perturbed_data[:, b] = add_noise(perturbed_data[:, b], noise_level)

                    perturbed_logits = model(perturbed_data)
                    mse_loss = F.mse_loss(original_logits, perturbed_logits, reduction='none').mean(dim=1)

                    for c in range(num_classes):
                        class_mask = (original_preds == c)
                        valid_mask = tp_mask & class_mask
                        band_importance[c, b] += mse_loss[valid_mask].sum()

                # 5. 새로운 방법 2️⃣: time 전체를 교란하여 중요도 측정
                for t in range(time):
                    perturbed_data = data.clone()
                    perturbed_data[:, :, t] = add_noise(perturbed_data[:, :, t], noise_level)

                    perturbed_logits = model(perturbed_data)
                    mse_loss = F.mse_loss(original_logits, perturbed_logits, reduction='none').mean(dim=1)

                    for c in range(num_classes):
                        class_mask = (original_preds == c)
                        valid_mask = tp_mask & class_mask
                        time_importance[c, t] += mse_loss[valid_mask].sum()

            # 반복 실험 결과 누적
            total_heatmap += heatmap
            total_band_importance += band_importance
            total_time_importance += time_importance

    # 평균 내기
    total_heatmap /= num_repeats
    total_band_importance /= num_repeats
    total_time_importance /= num_repeats

    # 히트맵 정규화 (기존 방식)
    for c in range(num_classes):
        if total_heatmap[c].sum() > 0:
            total_heatmap[c] /= total_heatmap[c].sum()
            total_heatmap[c] *= (band * time)

    # 새로운 방식 정규화 (band, time 각각 정규화)
    for c in range(num_classes):
        if total_band_importance[c].sum() > 0:
            total_band_importance[c] /= total_band_importance[c].sum()
            total_band_importance[c] *= band

        if total_time_importance[c].sum() > 0:
            total_time_importance[c] /= total_time_importance[c].sum()
            total_time_importance[c] *= time

    return (
        total_heatmap.cpu().numpy(),  # (num_classes, band, time)
        total_band_importance.cpu().numpy(),  # (num_classes, band)
        total_time_importance.cpu().numpy()  # (num_classes, time)
    )


def plot_psa_maps(importance_maps, band_importance, time_importance):
    """
    중요도 맵을 시각화하는 함수

    Args:
        importance_maps (np.array): (num_classes, num_bands, num_times) 형태의 중요도 맵
        band_importance (np.array): (num_classes, num_bands) 밴드별 중요도 (모든 값 교란)
        time_importance (np.array): (num_classes, num_times) 시기별 중요도 (모든 값 교란)
    """
    num_classes, num_bands, num_times = importance_maps.shape

    # 클래스별로 3개의 그래프 (히트맵, 교란된 밴드 중요도, 교란된 시기 중요도)
    fig, axes = plt.subplots(num_classes, 3, figsize=(18, 4 * num_classes))

    for cls in range(num_classes):
        class_name = target_name_mapping[cls]  # 클래스 이름 가져오기

        # 1️⃣ 시기별 & 밴드별 중요도 히트맵
        ax = axes[cls, 0]
        sns.heatmap(importance_maps[cls], cmap="coolwarm", annot=True, fmt=".2f", ax=ax)
        ax.set_title(f"{class_name}: Temporal & Band Importance")
        ax.set_xlabel("Time (Dates)")
        ax.set_ylabel("Bands (B, G, R, NIR)")
        ax.set_xticks(np.arange(num_times))
        ax.set_xticklabels(dates, rotation=45)  # 날짜 라벨 적용

        # 2️⃣ 새로운 방식: 교란된 밴드별 중요도 바 그래프
        ax = axes[cls, 1]
        ax.bar(["B", "G", "R", "NIR"], band_importance[cls], color=["blue", "green", "red", "purple"])
        ax.set_title(f"{class_name}: Band Importance")
        ax.set_ylabel("Importance Score")

        # 3️⃣ 새로운 방식: 교란된 시기별 중요도 바 그래프
        ax = axes[cls, 2]
        ax.bar(dates, time_importance[cls], color="darkorange")
        ax.set_title(f"{class_name}: Temporal Importance")
        ax.set_xlabel("Time (Dates)")
        ax.set_ylabel("Importance Score")
        ax.set_xticks(np.arange(num_times))
        ax.set_xticklabels(dates, rotation=45)  # 날짜 라벨 적용

    plt.tight_layout()
    plt.show()
    
#attention 분석
def compute_and_visualize_avg_attention(model, train_loader, num_classes, device):
    """
    1. TP(True Positive) 데이터 필터링
    2. 각 클래스별 Attention Score Map 누적 (첫 번째 / 마지막 / 전체 평균)
    3. 클래스별 평균 Attention Map 계산 후 시각화 (각 클래스당 3개)
    """
    model.eval()
    num_layers = None

    class_attention_first = {c: torch.zeros(12, 12).to(device) for c in range(num_classes)}
    class_attention_last = {c: torch.zeros(12, 12).to(device) for c in range(num_classes)}
    class_attention_avg = {c: torch.zeros(12, 12).to(device) for c in range(num_classes)}
    
    class_count = {c: 0 for c in range(num_classes)}

    with torch.no_grad():
        for x, y in tqdm(train_loader, desc="Processing Batches", unit="batch"):
            x, y = x.to(device), y.to(device)

            # ✅ 모델 Forward Pass (Attention Score 포함)
            logits, attention_weights = model(x, return_attention=True)

            # 🔹 attention_weights가 list인 경우 tensor로 변환
            if isinstance(attention_weights, list):
                attention_weights = torch.stack(attention_weights, dim=1)  # (batch, num_layers, heads, 12, 12)

            predictions = torch.argmax(logits, dim=1)  # 예측값 (batch,)

            if num_layers is None:
                num_layers = attention_weights.shape[1]

            # 🔹 TP 데이터 필터링 & Attention 누적
            for i in range(len(y)):
                true_label = y[i].item()
                pred_label = predictions[i].item()

                if true_label == pred_label:  # ✅ True Positive만 사용
                    
                    attn_first = attention_weights[i, 0].to(device)  # 첫 번째 레이어 평균 (12, 12)
                    attn_last = attention_weights[i, -1].to(device)  # 마지막 레이어 평균 (12, 12)
                    attn_avg = attention_weights[i].mean(dim=0).to(device)  # 전체 레이어 평균 (12, 12)
                    
                    class_attention_first[true_label] += attn_first
                    class_attention_last[true_label] += attn_last
                    class_attention_avg[true_label] += attn_avg
                    class_count[true_label] += 1

    # 🔹 클래스별 평균 Attention Map 계산 & 시각화
    for c in range(num_classes):
        if class_count[c] == 0:
            print(f"[경고] 클래스 {target_name_mapping[c]}의 TP 데이터가 없습니다. (스킵)")
            continue
        
        avg_first = class_attention_first[c] / class_count[c]
        avg_last = class_attention_last[c] / class_count[c]
        avg_all = class_attention_avg[c] / class_count[c]

        # 🔹 시각화
        for idx, (title, avg_attn) in enumerate([
            (f"Class {target_name_mapping[c]} - First Layer", avg_first),
            (f"Class {target_name_mapping[c]} - Last Layer", avg_last),
            (f"Class {target_name_mapping[c]} - All Layers Avg", avg_all),
        ]):
            plt.figure(figsize=(6, 5))
            sns.heatmap(avg_attn.cpu().numpy(), annot=False, cmap="viridis", square=True)
            plt.xlabel("Key Time Steps")
            plt.ylabel("Query Time Steps")
            plt.title(title)
            plt.show()

    print(f"✅ {num_classes * 3} 개의 평균 Attention Map이 출력되었습니다.")
