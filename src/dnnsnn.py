import warnings
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch_mlir.fx import export_and_import

# ==================================================
# [모델 정의]
# ==================================================
class RepeatedDNNSNNModel(nn.Module):
    def __init__(self, num_steps=10):
        super(RepeatedDNNSNNModel, self).__init__()
        self.num_steps = num_steps
        
        # 역전파를 위한 SNN 대체 그라디언트 설정
        spike_grad = surrogate.fast_sigmoid()
        
        # BLOCK 1: DNN 1 -> SNN 1
        self.dnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 출력 크기: [Batch, 16, 14, 14]
        )
        self.snn1_fc = nn.Linear(16 * 14 * 14, 64)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        
        # BLOCK 2: DNN 2 -> SNN 2
        self.dnn2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.snn2_fc = nn.Linear(32, 10)  # 최종 10개 클래스 분류
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

    def forward(self, x):
        # MLIR 추출을 위해 프로파일러용 record_function 블록은 제거하고 
        # 순수한 텐서 연산 흐름만 남겼습니다.
        
        batch_size = x.size(0)
        
        # [단계 1] 첫 번째 DNN 레이어 통과
        dnn1_out = self.dnn1(x)
        dnn1_flat = dnn1_out.view(batch_size, -1)
        snn1_input = self.snn1_fc(dnn1_flat)
        
        # [단계 2] 첫 번째 SNN 레이어 통과 (시간축 루프)
        mem1 = self.lif1.init_leaky()
        spk1_recording = []
        for step in range(self.num_steps):
            spk1, mem1 = self.lif1(snn1_input, mem1)
            spk1_recording.append(spk1)
        
        # [단계 3] 두 번째 DNN 레이어 통과 
        snn2_input_series = []
        for step in range(self.num_steps):
            current_spk1 = spk1_recording[step] 
            dnn2_out = self.dnn2(current_spk1)
            snn2_in = self.snn2_fc(dnn2_out)
            snn2_input_series.append(snn2_in)
            
        # [단계 4] 두 번째 SNN 레이어 통과
        mem2 = self.lif2.init_leaky()
        spk2_recording = []
        for step in range(self.num_steps):
            spk2, mem2 = self.lif2(snn2_input_series[step], mem2)
            spk2_recording.append(spk2)
                
        spk2_recording = torch.stack(spk2_recording)
        output_spikes = spk2_recording.sum(dim=0)
        
        return output_spikes

# ==================================================
# [MLIR 컴파일 실행 블록]
# ==================================================
if __name__ == "__main__":
    print("DNN+SNN 모델 초기화 및 준비 중...")
    
    # 1. 디바이스를 CPU로 설정하고, 모델을 평가(Eval) 모드로 전환
    model = RepeatedDNNSNNModel(num_steps=10).cpu()
    model.eval()  

    # 2. MLIR 추출을 위한 더미 입력 데이터 생성 (Batch: 4, Channel: 1, 28x28 Image)
    dummy_input = torch.randn(4, 1, 28, 28)

    print("MLIR Linalg 다이얼렉트로 컴파일 중...")

    # 3. 최신 함수인 export_and_import를 사용하여 변환
    with torch.no_grad():
        mlir_module = export_and_import(
            model,
            dummy_input,
            output_type="linalg-on-tensors" # 고수준 수학 연산 형태의 다이얼렉트 지정
        )

    # 4. 생성된 MLIR 코드를 파일로 저장
    output_filename = "outputs/snn_dnn_linalg.mlir"
    with open(output_filename, "w") as f:
        f.write(str(mlir_module))

    print(f"변환 성공! '{output_filename}' 파일이 생성되었습니다.")
