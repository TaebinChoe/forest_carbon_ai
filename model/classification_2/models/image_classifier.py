
import torch
import torch.nn as nn

"""
<input_shape>
channel: 48, patch_size: 3
"""
class BasicBlock_48_3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_48_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetLike_48_3(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetLike_48_3, self).__init__()
        self.in_channels = 64
        
        # 48 채널 입력을 처리할 Conv Layer
        self.conv1 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=1)
        self.layer3 = self._make_layer(256, 2, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock_48_3(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock_48_3(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

"""
<input_shape>
channel: 60, patch_size: 3
"""
class BasicBlock_60_3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_60_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetLike_60_3(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetLike_60_3, self).__init__()
        self.in_channels = 64
        
        # 60 채널 입력을 처리할 Conv Layer
        self.conv1 = nn.Conv2d(60, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=1)
        self.layer3 = self._make_layer(256, 2, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock_60_3(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock_60_3(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

"""
<input_shape>
channel: 108, patch_size:3
"""
class BasicBlock_108_3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_108_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetLike_108_3(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetLike_108_3, self).__init__()
        self.in_channels = 128  # 초기 채널 수 설정
        
        # 108 채널 입력을 처리할 Conv Layer
        self.conv1 = nn.Conv2d(108, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        
        # 블록 채널 증가: 128 → 256 → 512
        self.layer1 = self._make_layer(128, 2, stride=1)
        self.layer2 = self._make_layer(256, 2, stride=1)
        self.layer3 = self._make_layer(512, 2, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock_108_3(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock_108_3(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    
"""
<input_shape>
channel: 120, patch_size:3
"""
class BasicBlock_120_3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_120_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetLike_120_3(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNetLike_120_3, self).__init__()
        self.in_channels = 128  # 128로 증가

        # 120 채널 입력을 처리할 Conv Layer
        self.conv1 = nn.Conv2d(120, 128, kernel_size=3, stride=1, padding=1, bias=False)  # 64 → 128로 변경
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        
        # 블록 채널 증가: 128 → 256 → 512
        self.layer1 = self._make_layer(128, 2, stride=1)
        self.layer2 = self._make_layer(256, 2, stride=1)
        self.layer3 = self._make_layer(512, 2, stride=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)  # 최종 Fully Connected Layer도 512로 변경
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock_120_3(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock_120_3(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

"""
<input_shape>
channel: 48, patch_size: variable
"""
class BasicBlock_48(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_48, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNetLike_48(nn.Module):
    def __init__(self, num_classes=6, patch_size=1):
        super(ResNetLike_48, self).__init__()
        self.in_channels = 64
        self.patch_size = patch_size
        
        # 입력 크기에 맞게 조정 (48, patch_size, patch_size)
        self.conv1 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=1)
        self.layer3 = self._make_layer(256, 2, stride=1)
        
        # AdaptiveAvgPool 크기 조정
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock_48(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock_48(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 기본 블록: 입력과 출력의 형태가 동일하며, C(채널) 차원에서 연산을 집중적으로 수행
class BasicBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super(BasicBlock, self).__init__()
        self.norm = nn.BatchNorm2d(channels)  # 채널별 정규화 수행
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)  # 채널 유지, 패딩 적용
        self.dropout = nn.Dropout(dropout)  # 일부 뉴런을 무작위로 끄는 드롭아웃
        self.activation = nn.SiLU()  # Swish 활성화 함수 사용

    def forward(self, x):
        identity = x  # Skip Connection (잔차 연결)
        out = self.norm(x)  # 정규화 수행
        out = self.conv(out)  # 합성곱 연산
        out = self.dropout(out)  # 드롭아웃 적용
        out = self.activation(out)  # 활성화 함수 적용
        out += identity  # 입력과 출력 더하기 (잔차 연결)
        return out

# 두께 늘림 블록: 채널 수를 동적으로 증가시키면서 정보를 확장하는 역할 수행
class DepthExpansionBlock(nn.Module):
    def __init__(self, in_channels, expansion_factor=2):
        super(DepthExpansionBlock, self).__init__()
        out_channels = round(in_channels * expansion_factor)  # 확장된 채널 수를 정수로 반올림
        self.norm = nn.BatchNorm2d(in_channels)  # 입력 채널 정규화
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 채널 확장, 공간 정보는 유지
        self.activation = nn.ReLU()  # 활성화 함수 사용
        self.out_channels = out_channels  # 확장된 채널 수 저장

    def forward(self, x):
        out = self.norm(x)  # 정규화 수행
        out = self.conv(out)  # 1x1 컨볼루션으로 채널 수 확장
        out = self.activation(out)  # 활성화 함수 적용
        return out

# 피드포워드 블록: 채널별 연산을 수행하며, 입력과 출력의 형태가 동일함
class FeedForwardBlock(nn.Module):
    def __init__(self, channels, expansion=4, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.norm = nn.BatchNorm2d(channels)  # 채널 정규화
        hidden_dim = channels * expansion  # 확장된 채널 수 설정
        self.fc1 = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)  # 채널 확장
        self.activation = nn.SiLU()  # Swish 활성화 함수 적용
        self.dropout = nn.Dropout(dropout)  # 드롭아웃 적용
        self.fc2 = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)  # 원래 채널로 축소

    def forward(self, x):
        identity = x  # Skip Connection (잔차 연결)
        out = self.norm(x)  # 정규화 수행
        out = self.fc1(out)  # 채널 확장
        out = self.activation(out)  # 활성화 함수 적용
        out = self.dropout(out)  # 드롭아웃 적용
        out = self.fc2(out)  # 채널 축소하여 원래 차원으로 복귀
        out += identity  # 입력과 출력 더하기 (잔차 연결)
        return out

# 전체 모델 구조 정의
class ResNetLike(nn.Module):
    def __init__(self, input_channels=48, num_classes=6):
        super(ResNetLike, self).__init__()
        self.initial_norm = nn.BatchNorm2d(input_channels)  # 초기 입력 데이터 정규화
        self.initial_conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, bias=False)  # 초기 컨볼루션
        self.initial_activation = nn.ReLU()  # 활성화 함수
        
        self.layer1 = BasicBlock(input_channels)  # 기본 블록 적용 (입력과 출력 형태 동일)
        self.layer2 = DepthExpansionBlock(input_channels, expansion_factor=2)  # 채널 2배 확장
        expanded_c1 = self.layer2.out_channels  # 확장된 채널 수
        
        self.layer3 = BasicBlock(expanded_c1)  # 기본 블록 적용
        self.layer4 = DepthExpansionBlock(expanded_c1, expansion_factor=1.5)  # 채널 1.5배 확장
        expanded_c2 = self.layer4.out_channels  # 확장된 채널 수
        
        self.layer5 = FeedForwardBlock(expanded_c2)  # 피드포워드 블록 적용
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 글로벌 평균 풀링
        self.fc = nn.Linear(expanded_c2, num_classes)  # 최종 분류기
    
    def forward(self, x):
        x = self.initial_norm(x)  # 초기 정규화 적용
        x = self.initial_conv(x)  # 첫 번째 컨볼루션 적용
        x = self.initial_activation(x)  # 활성화 함수 적용
        
        x = self.layer1(x)  # 기본 블록 적용
        x = self.layer2(x)  # 두께 늘림 블록 적용
        x = self.layer3(x)  # 기본 블록 적용
        x = self.layer4(x)  # 두께 늘림 블록 적용
        x = self.layer5(x)  # 피드포워드 블록 적용
        
        x = self.avgpool(x)  # 풀링 적용
        x = torch.flatten(x, 1)  # 차원 축소
        x = self.fc(x)  # 최종 분류기 적용
        return x
