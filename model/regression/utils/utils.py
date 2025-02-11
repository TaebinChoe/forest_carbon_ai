import os
import numpy as np
import tifffile as tiff
from torch.utils.data import Dataset

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