# import torch
# import torch.nn as nn

# """
# video 처럼 처리하는 모델델
# """
# # SwigLU 활성 함수 정의 (SiLU 변형)
# class SwigLU(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(1.3 * x)

# # 2+1D 컨볼루션 블록: 공간과 시간 축을 분리하여 컨볼루션 수행
# class Conv2Plus1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=1):
#         super().__init__()
#         mid_channels = out_channels  # 중간 채널 수 설정
        
#         # 공간 차원에 대한 2D 컨볼루션
#         self.spatial_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=(1, kernel_size[1], kernel_size[2]), 
#                                       stride=(1, stride, stride), padding=(0, padding, padding), bias=False)
#         self.bn1 = nn.BatchNorm3d(mid_channels)
        
#         # 시간 차원에 대한 1D 컨볼루션
#         self.temporal_conv = nn.Conv3d(mid_channels, out_channels, kernel_size=(kernel_size[0], 1, 1), 
#                                        stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False)
#         self.bn2 = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         x = self.relu(self.bn1(self.spatial_conv(x)))
#         x = self.relu(self.bn2(self.temporal_conv(x)))
#         return x

# # ResNet 스타일의 기본 블록 (Skip Connection 포함)
# class ResBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = Conv2Plus1D(channels, channels)
#         self.conv2 = Conv2Plus1D(channels, channels)
#         self.bn = nn.BatchNorm3d(channels)
#         self.relu = nn.ReLU(inplace=True)
    
#     def forward(self, x):
#         residual = x  # 원본 입력 저장
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.bn(x)
#         return self.relu(x + residual)  # Skip Connection 적용

# # 밴드(채널) 확장 블록: 1x1 컨볼루션 사용
# class BandExpansion(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn = nn.BatchNorm3d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
    
#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

# # 피드포워드 블록: SwigLU 활성 함수와 확장 비율 적용
# class FeedForward(nn.Module):
#     def __init__(self, channels, expand_ratio=2):
#         super().__init__()
#         hidden_dim = channels * expand_ratio  # 확장된 채널 크기
#         self.fc1 = nn.Conv3d(channels, hidden_dim, kernel_size=1)
#         self.act = SwigLU()
#         self.fc2 = nn.Conv3d(hidden_dim, channels, kernel_size=1)
#         self.bn = nn.BatchNorm3d(channels)
    
#     def forward(self, x):
#         return self.bn(self.fc2(self.act(self.fc1(x))))

# # 전체 비디오 분류 모델
# class VideoClassifier(nn.Module):
#     def __init__(self, input_bands, stage_repeats, stage_channels, num_classes=6):
#         super().__init__()
        
#         # 초기 밴드 확장 (입력 밴드 수 -> 첫 번째 stage 채널 크기)
#         self.initial_conv = BandExpansion(input_bands, stage_channels[0])
        
#         self.stages = nn.ModuleList()
#         in_channels = stage_channels[0]
        
#         # 4개의 Stage 구성
#         for stage_idx in range(4):
#             # 기본 ResBlock 반복 적용
#             blocks = [ResBlock(in_channels) for _ in range(stage_repeats[stage_idx])]
#             self.stages.append(nn.Sequential(*blocks))
            
#             # 마지막 Stage를 제외하고 밴드 확장 수행
#             if stage_idx < 3:
#                 self.stages.append(BandExpansion(in_channels, stage_channels[stage_idx + 1]))
#                 in_channels = stage_channels[stage_idx + 1]
        
#         # 글로벌 평균 풀링 적용 (공간 차원 제거)
#         self.gap = nn.AdaptiveAvgPool3d(1)
        
#         # 피드포워드 블록 2개 추가
#         self.ff1 = FeedForward(in_channels)
#         self.ff2 = FeedForward(in_channels)
        
#         # 최종 분류기 (Linear 레이어)
#         self.classifier = nn.Linear(in_channels, num_classes)
        
#     def forward(self, x):
#         x = self.initial_conv(x)  # 초기 밴드 확장 적용
        
#         # Stage 반복 수행
#         for stage in self.stages:
#             x = stage(x)
        
#         # 글로벌 평균 풀링 적용 후 차원 축소
#         x = self.gap(x).view(x.size(0), -1)
        
#         # 피드포워드 블록 두 개 통과
#         x = self.ff1(x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
#         x = self.ff2(x)
        
#         # 최종 분류기 적용
#         x = self.classifier(x.squeeze(-1).squeeze(-1).squeeze(-1))
#         return x

import torch
import torch.nn as nn
import torch.nn.init as init

"""
비디오 데이터를 (Batch, Channels, Temp, Height, Width) 형식으로 입력받아 처리하는 모델
시간 차원의 중요성을 반영하기 위해 Conv2Plus1D에서 시간 축을 먼저 처리한 후 공간 정보를 학습함
Dropout을 추가하여 모델의 일반화 성능을 높임
"""

# SwigLU 활성 함수 정의 (SiLU 변형)
class SwigLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.3 * x)

# 2+1D 컨볼루션 블록: 시간 축을 먼저 처리한 후 공간 차원을 학습
class Conv2Plus1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), stride=1, padding=1, dropout=0.3):
        super().__init__()
        mid_channels = out_channels  # 중간 채널 수 설정
        
        # 시간 차원에 대한 1D 컨볼루션
        self.temporal_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=(kernel_size[0], 1, 1), 
                                       stride=(stride, 1, 1), padding=(padding, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout3d(dropout)
        
        # 공간 차원에 대한 2D 컨볼루션
        self.spatial_conv = nn.Conv3d(mid_channels, out_channels, kernel_size=(1, kernel_size[1], kernel_size[2]), 
                                      stride=(1, stride, stride), padding=(0, padding, padding), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout2 = nn.Dropout3d(dropout)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.temporal_conv(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.spatial_conv(x)))
        x = self.dropout2(x)
        return x

# ResNet 스타일의 기본 블록 (Skip Connection 포함)
class ResBlock(nn.Module):
    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.conv1 = Conv2Plus1D(channels, channels, dropout=dropout)
        self.conv2 = Conv2Plus1D(channels, channels, dropout=dropout)
        self.bn = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x  # 원본 입력 저장
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x + residual)  # Skip Connection 적용

# 밴드(채널) 확장 블록: 1x1 컨볼루션 사용
class BandExpansion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# 피드포워드 블록: SwigLU 활성 함수와 확장 비율 적용
class FeedForward(nn.Module):
    def __init__(self, channels, expand_ratio=2, dropout=0.3):
        super().__init__()
        hidden_dim = channels * expand_ratio  # 확장된 채널 크기
        self.fc1 = nn.Conv3d(channels, hidden_dim, kernel_size=1)
        self.act = SwigLU()
        self.fc2 = nn.Conv3d(hidden_dim, channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout3d(dropout)
    
    def forward(self, x):
        return self.dropout(self.bn(self.fc2(self.act(self.fc1(x)))))

# 전체 비디오 분류 모델
class VideoClassifier(nn.Module):
    def __init__(self, input_bands, stage_repeats, stage_channels, num_classes=6, dropout=0.3):
        super().__init__()
        
        # 초기 밴드 확장 (입력 밴드 수 -> 첫 번째 stage 채널 크기)
        self.initial_conv = BandExpansion(input_bands, stage_channels[0])
        
        self.stages = nn.ModuleList()
        in_channels = stage_channels[0]
        
        # 4개의 Stage 구성
        for stage_idx in range(4):
            # 기본 ResBlock 반복 적용
            blocks = [ResBlock(in_channels, dropout) for _ in range(stage_repeats[stage_idx])]
            self.stages.append(nn.Sequential(*blocks))
            
            # 마지막 Stage를 제외하고 밴드 확장 수행
            if stage_idx < 3:
                self.stages.append(BandExpansion(in_channels, stage_channels[stage_idx + 1]))
                in_channels = stage_channels[stage_idx + 1]
        
        # 글로벌 평균 풀링 적용 (공간 차원 제거)
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # 피드포워드 블록 2개 추가
        self.ff1 = FeedForward(in_channels, dropout=dropout)
        self.ff2 = FeedForward(in_channels, dropout=dropout)
        
        # 최종 분류기 (Linear 레이어)
        self.classifier = nn.Linear(in_channels, num_classes)
        
        # 가중치 초기화 적용
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.initial_conv(x)  # 초기 밴드 확장 적용
        
        # Stage 반복 수행
        for stage in self.stages:
            x = stage(x)
        
        # 글로벌 평균 풀링 적용 후 차원 축소
        x = self.gap(x).view(x.size(0), -1)
        
        # 피드포워드 블록 두 개 통과
        x = self.ff1(x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        x = self.ff2(x)
        
        # 최종 분류기 적용
        x = self.classifier(x.squeeze(-1).squeeze(-1).squeeze(-1))
        return x
