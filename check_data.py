import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 시각화 설정을 위한 매직 커맨드 (주피터용)
# %matplotlib inline 

print("라이브러리 로드 완료!")

# --- [RadarDataset 클래스 정의] ---
class RadarDataset(Dataset):
    def __init__(self, mat_file_path):
        """
        mat_file_path: .mat 파일의 경로
        """
        self.file_path = mat_file_path
        
        # 데이터 로드 (메모리에 올리기)
        try:
            with h5py.File(self.file_path, 'r') as f:
                # 1. 데이터 읽기 및 전치 (.T) -> (N, Length)
                X_raw = np.array(f['X_noisy']).T
                Y_raw = np.array(f['Y_clean']).T
                
                # 2. 복소수 변환 (구조체 -> complex128)
                # MATLAB v7.3 이슈 해결 코드
                if X_raw.dtype.names is not None and 'real' in X_raw.dtype.names:
                    self.X_data = X_raw['real'] + 1j * X_raw['imag']
                    self.Y_data = Y_raw['real'] + 1j * Y_raw['imag']
                else:
                    self.X_data = X_raw
                    self.Y_data = Y_raw
                    
            print(f"데이터셋 로드 완료: {len(self.X_data)} 샘플")
            
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다: {mat_file_path}")
            self.X_data = []
            self.Y_data = []

    def __len__(self):
        # 데이터의 총 개수 반환
        return len(self.X_data)

    def __getitem__(self, idx):
        # 인덱스(idx)에 해당하는 데이터 하나를 가져오는 함수
        
        # 1. 데이터 하나 꺼내기
        x_complex = self.X_data[idx] # (1024,) 복소수
        y_complex = self.Y_data[idx] # (1024,) 복소수
        
        # 2. 딥러닝 입력용 변환: (Real, Imag) 2채널로 분리
        # 결과 형태: [2, 1024]
        # np.stack을 써서 두 줄로 쌓습니다.
        x_2ch = np.stack([np.real(x_complex), np.imag(x_complex)], axis=0)
        y_2ch = np.stack([np.real(y_complex), np.imag(y_complex)], axis=0)
        
        # 3. PyTorch Tensor로 변환 (float32 사용)
        x_tensor = torch.from_numpy(x_2ch).float()
        y_tensor = torch.from_numpy(y_2ch).float()
        
        return x_tensor, y_tensor

# --- [실행 및 테스트] ---
if __name__ == "__main__":
    # 1. 데이터셋 객체 생성 (파일 경로는 실제 환경에 맞게 수정 필요)
    # 예: 'Radar_Denoising/sine_dataset.mat'
    file_path = 'Radar_Denoising/sine_dataset.mat'
    
    # 파일이 실제로 있을 때만 실행
    import os
    if os.path.exists(file_path):
        dataset = RadarDataset(file_path)

        # 2. 데이터로더(DataLoader) 생성
        # 배치(Batch): 데이터를 한 번에 32개씩 모델에 전달
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 3. 첫 번째 배치를 꺼내서 모양 확인
        data_iter = iter(dataloader)
        inputs, labels = next(data_iter)

        print("-" * 30)
        print(f"총 데이터 개수: {len(dataset)}")
        print(f"배치 사이즈(Batch Size): {inputs.shape[0]}")
        print(f"입력 데이터 형태 (Input Shape): {inputs.shape}") # 예상: [32, 2, 1024]
        print(f"정답 데이터 형태 (Label Shape): {labels.shape}") # 예상: [32, 2, 1024]
        print("-" * 30)
    else:
        print(f"경고: '{file_path}' 파일이 없습니다. 경로를 확인해주세요.")
