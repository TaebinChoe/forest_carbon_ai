import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch
import tifffile as tiff
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import copy
import torch.nn as nn
import torch.nn.init as init
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device 설정

#train 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = model.state_dict()  # 초기 모델 가중치 저장
    no_improve_count = 0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)  # 배치별 loss * 개수로 전체 손실 계산
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)  # 전체 샘플 수로 나눔
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)

        print(f"\nEpoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")

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

#eval함수

# 클래스별 라벨 이름 정의
target_name_mapping = {
    0: "Non-Forest",  # 비산림
    1: "Pine",  # 소나무
    2: "Nut Pine",  # 잣나무
    3: "Larch",  # 낙엽송
    4: "Mongolian Oak",  # 신갈나무
    5: "Oriental Oak"  # 굴참나무
}

# 모델 평가 함수
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
    report_df["Category"] = "Overall"
    
    # 추가적인 분석 수행 및 결과 저장
    conifer_vs_broadleaf_report = evaluate_conifer_vs_broadleaf(all_labels, all_predictions) #침/활 분류력
    conifer_vs_broadleaf_report["Category"] = "Conifer vs Broadleaf"
    
    conifer_report = evaluate_conifer_classification(all_labels, all_predictions) #침엽수 내 분류력
    conifer_report["Category"] = "Conifer"
    
    broadleaf_report = evaluate_broadleaf_classification(all_labels, all_predictions) #활엽수 내 분류력
    broadleaf_report["Category"] = "Broadleaf"
    
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
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pine", "Nut Pine", "Larch"], yticklabels=["Pine", "Nut Pine", "Larch"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Conifer Classification')
    plt.show()
    
    return pd.DataFrame(classification_report(true_labels, pred_labels, target_names=["Pine", "Nut Pine", "Larch"], digits=3, output_dict=True)).transpose()

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
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Mongolian Oak", "Oriental Oak"], yticklabels=["Mongolian Oak", "Oriental Oak"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Broadleaf Classification')
    plt.show()
    
    return pd.DataFrame(classification_report(true_labels, pred_labels, target_names=["Mongolian Oak", "Oriental Oak"], digits=3, output_dict=True)).transpose()

#rf평가를 위함
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm

# 클래스별 라벨 이름 정의
target_name_mapping = {
    0: "Non-Forest",  # 비산림
    1: "Pine",  # 소나무
    2: "Nut Pine",  # 잣나무
    3: "Larch",  # 낙엽송
    4: "Mongolian Oak",  # 신갈나무
    5: "Oriental Oak"  # 굴참나무
}

# 모델 평가 함수
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
    report_df["Category"] = "Overall"
    
    # 추가적인 분석 수행 및 결과 저장
    conifer_vs_broadleaf_report = evaluate_conifer_vs_broadleaf(all_labels, all_predictions)  # 침/활 분류력
    conifer_vs_broadleaf_report["Category"] = "Conifer vs Broadleaf"
    
    conifer_report = evaluate_conifer_classification(all_labels, all_predictions)  # 침엽수 내 분류력
    conifer_report["Category"] = "Conifer"
    
    broadleaf_report = evaluate_broadleaf_classification(all_labels, all_predictions)  # 활엽수 내 분류력
    broadleaf_report["Category"] = "Broadleaf"
    
    # 결과를 하나의 데이터프레임으로 통합
    additional_metrics = pd.concat([conifer_vs_broadleaf_report, conifer_report, broadleaf_report])
    final_report_df = pd.concat([report_df, additional_metrics])
    
    return final_report_df


#dataset class
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

#sdi 함수
def sdi_importance_analysis(model, data_loader, num_samples=1, perturbation_strength=0.2, target_dims=None):
    model.eval()

    sample_batch, _ = next(iter(data_loader))
    num_dims = len(sample_batch.shape) - 1
    dim_names = [f"dim_{i}" for i in range(1, num_dims + 1)]

    # 조사할 차원을 설정 (기본: 모든 차원)
    if target_dims is None:
        target_dims = dim_names
    else:
        target_dims = [f"dim_{i}" for i in target_dims]

    importance_scores = {dim: 0.0 for dim in target_dims}
    per_class_scores = {}

    num_batches = 0
    for X_batch, _ in data_loader:
        num_batches += 1
        X_batch = X_batch.to(next(model.parameters()).device)

        logit_original = model(X_batch).detach()
        num_classes = logit_original.shape[1]

        if not per_class_scores:
            per_class_scores = {cls: {dim: 0.0 for dim in target_dims} for cls in range(num_classes)}

        for dim_name in target_dims:
            dim_idx = dim_names.index(dim_name) + 1  # 1-based index
            total_mse = 0.0
            class_mse = {cls: 0.0 for cls in range(num_classes)}

            for _ in range(num_samples):
                X_perturbed = X_batch.clone().detach()
                num_swap = max(1, int(X_perturbed.shape[dim_idx] * perturbation_strength))
                swap_indices = random.sample(range(X_perturbed.shape[dim_idx]), num_swap)
                permutation = random.sample(swap_indices, len(swap_indices))

                X_perturbed.index_copy_(dim_idx, torch.tensor(swap_indices, device=X_batch.device),
                                        X_perturbed.index_select(dim_idx, torch.tensor(permutation, device=X_batch.device)))

                logit_perturbed = model(X_perturbed).detach()
                mse = torch.mean((logit_original - logit_perturbed) ** 2, dim=0)
                total_mse += mse.mean().item()

                for cls in range(num_classes):
                    class_mse[cls] += mse[cls].item()

            importance_scores[dim_name] += total_mse / num_samples
            for cls in range(num_classes):
                per_class_scores[cls][dim_name] += class_mse[cls] / num_samples

    for key in importance_scores:
        importance_scores[key] /= num_batches

    for cls in per_class_scores:
        for key in per_class_scores[cls]:
            per_class_scores[cls][key] /= num_batches

    # 🔹 선택한 차원만 정규화하여 합이 1이 되도록 조정
    total_score = sum(importance_scores.values())
    importance_scores = {key: value / total_score for key, value in importance_scores.items()} if total_score > 0 else importance_scores

    for cls in per_class_scores:
        class_total_score = sum(per_class_scores[cls].values())
        per_class_scores[cls] = {key: value / class_total_score for key, value in per_class_scores[cls].items()} if class_total_score > 0 else per_class_scores[cls]

    return {"overall": importance_scores, "per_class": per_class_scores}


def plot_importance_scores(importance_scores, per_class_scores):
    # Overall Dimension Importance (Bar Chart)
    plt.figure(figsize=(8, 5))
    plt.bar(importance_scores.keys(), importance_scores.values(), color='skyblue')
    plt.xlabel("Dimension")
    plt.ylabel("Importance Score")
    plt.title("Overall Dimension Importance")
    plt.xticks(rotation=45)
    plt.show()

    # Per-Class Importance (Heatmap)
    per_class_df = {cls: list(scores.values()) for cls, scores in per_class_scores.items()}
    dim_labels = list(per_class_scores[0].keys())  # Dimension names

    plt.figure(figsize=(10, 6))
    sns.heatmap(list(per_class_df.values()), annot=True, cmap="Blues", xticklabels=dim_labels, yticklabels=list(per_class_df.keys()))
    plt.xlabel("Dimension")
    plt.ylabel("Class")
    plt.title("Per-Class Dimension Importance")
    plt.show()