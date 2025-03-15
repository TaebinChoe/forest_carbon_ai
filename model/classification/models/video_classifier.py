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
import torch.nn.functional as F
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
#(Token_length, embedding) = (12, 4×81)인 transformer 모델

class TransformerModel_(nn.Module):
    def __init__(self, patch_size=9, num_bands=4, temp=12, num_classes=6, 
                 d_model=64, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.temp = temp
        self.num_tokens = temp  # 12개 (시계열 단위)
        self.d_model = d_model  

        # ** 입력 차원 변환 (4×9×9 → d_model) **
        self.input_projection = nn.Linear(num_bands * patch_size * patch_size, d_model)

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

        # ** Layer Normalization 추가 **
        self.norm1 = nn.LayerNorm(d_model)  

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,  
            dim_feedforward=dim_feedforward,  
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head (Dropout 추가)
        self.fc = nn.Sequential(
            nn.Linear(self.num_tokens * d_model, 128),  # 192 → 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  # 96 → 64
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        # ** Layer Normalization 추가 **
        self.norm2 = nn.LayerNorm(num_classes)  

        # ** 가중치 초기화 **
        self.apply(self._init_weights)  

    def forward(self, x):
        batch_size = x.shape[0]

        # (batch, 4, 12, 9, 9) → (batch, 12, 4×9×9)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size, self.num_tokens, -1)

        # 입력 차원 변환 (4×9×9 → d_model)
        x = self.input_projection(x)

        # Add Positional Encoding
        x = x + self.pos_embedding  

        # Transformer Encoder
        x = self.norm1(x)  
        x = self.transformer(x)  

        # Flatten & Classification
        x = x.flatten(1)  
        x = self.fc(x)

        # 출력 정규화
        x = self.norm2(x)  

        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)  
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

#새로운 transformer 모델
import torch
import torch.nn as nn

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=False, **kwargs):
        """Residual Connection 후 LayerNorm 적용"""
        
        # Self-Attention Layer
        src2, attn_weights = self.self_attn(
            src, src, src, attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask, need_weights=True
        )  
        src = self.norm1(src + self.dropout1(src2))  # 🔹 Residual 후 LayerNorm
        
        # Feedforward Layer
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))  # 🔹 Residual 후 LayerNorm
        
        if return_attention:
            return src, attn_weights
        return src

class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, num_layers=4, nhead=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, return_attention=False):
        if return_attention:
            outputs, attn_weights = [], []
            for layer in self.encoder.layers:
                src, attn = layer(src, return_attention=True)
                attn_weights.append(attn)
                outputs.append(src)

            return outputs[-1], torch.stack(attn_weights, dim=1)  
        return self.encoder(src)

class TransformerModel(nn.Module):
    def __init__(self, patch_size=9, num_bands=4, temp=12, num_classes=6, 
                 d_model=64, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.temp = temp
        self.num_tokens = temp
        self.d_model = d_model  

        # **입력 차원 변환 후 LayerNorm 추가**
        self.input_projection = nn.Linear(num_bands * patch_size * patch_size, d_model)
        self.norm_input = nn.LayerNorm(d_model)  # 🔹 입력 정규화 추가

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

        # Transformer Encoder
        self.transformer = CustomTransformerEncoder(d_model, num_layers, nhead, dim_feedforward, dropout)

        # Classification Head (Dropout 강화)
        self.fc = nn.Sequential(
            nn.Linear(self.num_tokens * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.norm_out = nn.LayerNorm(num_classes)  # 🔹 출력 정규화 추가

        # **가중치 초기화**
        self.apply(self._init_weights)  

    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]

        # (batch, 4, 12, 9, 9) → (batch, 12, 4×9×9)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size, self.num_tokens, -1)

        # 입력 차원 변환 & LayerNorm 적용
        x = self.input_projection(x)
        x = self.norm_input(x)  # 🔹 입력 정규화

        # Positional Encoding 추가
        x = x + self.pos_embedding  

        # Transformer Encoder
        if return_attention:
            x, attn_weights = self.transformer(x, return_attention=True)
        else:
            x = self.transformer(x)

        # Flatten & Classification
        x = x.flatten(1)  
        x = self.fc(x)

        # 출력 정규화
        x = self.norm_out(x)  

        if return_attention:
            return x, attn_weights  
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)  # 🔹 Xavier Normal 적용
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

# 🔹 Gradient Clipping 적용 (훈련 루프에서 추가)
def train_model(model, dataloader, optimizer, criterion, clip_value=1.0):
    model.train()
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)  # 🔹 Gradient Clipping
        optimizer.step()

# class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
#     def forward(self, src, src_mask=None, src_key_padding_mask=None, return_attention=False, **kwargs):
#         """Attention 가중치를 반환하도록 수정 + is_causal 인자 무시"""
        
#         # `is_causal` 같은 추가 인자를 무시하도록 kwargs 처리
#         src2, attn_weights = self.self_attn(
#             src, src, src, attn_mask=src_mask, 
#             key_padding_mask=src_key_padding_mask, need_weights=True
#         )  
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
        
#         if return_attention:
#             return src, attn_weights  # 🔹 Attention 가중치 반환
#         else:
#             return src

# class CustomTransformerEncoder(nn.Module):
#     def __init__(self, d_model=64, num_layers=4, nhead=4, dim_feedforward=128, dropout=0.1):
#         super().__init__()
#         encoder_layer = CustomTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, src, return_attention=False):
#         """Custom Transformer Encoder: Attention 가중치를 반환할 수 있도록 수정"""
#         if return_attention:
#             outputs, attn_weights = [], []
#             for layer in self.encoder.layers:
#                 src, attn = layer(src, return_attention=True)
#                 attn_weights.append(attn)
#                 outputs.append(src)

#             return outputs[-1], torch.stack(attn_weights, dim=1)  # (batch, num_layers, seq_len, seq_len)
#         else:
#             return self.encoder(src)

# class TransformerModel(nn.Module):
#     def __init__(self, patch_size=9, num_bands=4, temp=12, num_classes=6, 
#                  d_model=64, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1):
#         super().__init__()
#         self.patch_size = patch_size
#         self.num_bands = num_bands
#         self.temp = temp
#         self.num_tokens = temp  # 12개 (시계열 단위)
#         self.d_model = d_model  

#         # ** 입력 차원 변환 (4×9×9 → d_model) **
#         self.input_projection = nn.Linear(num_bands * patch_size * patch_size, d_model)

#         # Positional Encoding
#         self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

#         # ** Layer Normalization 추가 **
#         self.norm1 = nn.LayerNorm(d_model)  

#         # Transformer Encoder (Custom 버전 적용)
#         self.transformer = CustomTransformerEncoder(d_model, num_layers, nhead, dim_feedforward, dropout)

#         # Classification Head (Dropout 추가)
#         self.fc = nn.Sequential(
#             nn.Linear(self.num_tokens * d_model, 128),  # 192 → 128
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),  # 96 → 64
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )

#         # ** Layer Normalization 추가 **
#         self.norm2 = nn.LayerNorm(num_classes)  

#         # ** 가중치 초기화 **
#         self.apply(self._init_weights)  

#     def forward(self, x, return_attention=False):
#         batch_size = x.shape[0]

#         # (batch, 4, 12, 9, 9) → (batch, 12, 4×9×9)
#         x = x.permute(0, 2, 1, 3, 4).reshape(batch_size, self.num_tokens, -1)

#         # 입력 차원 변환 (4×9×9 → d_model)
#         x = self.input_projection(x)

#         # Add Positional Encoding
#         x = x + self.pos_embedding  

#         # Transformer Encoder
#         x = self.norm1(x)  
#         if return_attention:
#             x, attn_weights = self.transformer(x, return_attention=True)
#         else:
#             x = self.transformer(x)

#         # Flatten & Classification
#         x = x.flatten(1)  
#         x = self.fc(x)

#         # 출력 정규화
#         x = self.norm2(x)  

#         if return_attention:
#             return x, attn_weights  # 🔹 Attention 가중치도 반환
#         return x

#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.xavier_uniform_(module.weight)  
#             if module.bias is not None:
#                 nn.init.constant_(module.bias, 0)

                
#hybrid 모델            
class HybridCNNTransformer(nn.Module):
    def __init__(self, patch_size=9, num_bands=4, temp=12, num_classes=6, 
                 d_model=64, nhead=4, num_layers=4, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.temp = temp
        self.num_tokens = temp  # 12개 (시계열 단위)
        self.d_model = d_model  

        # ** 2+1D CNN Feature Extractor **
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels=num_bands, out_channels=32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=d_model, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((self.num_tokens, 1, 1))  # (B, d_model, 12, 1, 1)
        )

        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model))

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.fc = nn.Sequential(
            nn.Linear(self.num_tokens * d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

        self.norm2 = nn.LayerNorm(num_classes)

        self.apply(self._init_weights)

    def forward(self, x):
        batch_size = x.shape[0]

        # CNN Feature Extraction (B, 4, 12, 9, 9) -> (B, d_model, 12, 1, 1)
        x = self.cnn_feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)  # (B, d_model, 12)
        x = x.permute(0, 2, 1)  # (B, 12, d_model)

        # Add Positional Encoding
        x = x + self.pos_embedding

        # Transformer Encoder
        x = self.norm1(x)
        x = self.transformer(x)

        # Flatten & Classification
        x = x.flatten(1)
        x = self.fc(x)
        x = self.norm2(x)

        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

