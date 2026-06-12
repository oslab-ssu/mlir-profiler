# SNN-DNN MLIR Profiling Pipeline

이 프로젝트는 PyTorch 기반의 하이브리드 SNN-DNN 모델을 MLIR(Linalg, LLVM Dialect)을 거쳐 LLVM IR로 하향(Lowering)하고, 최종적으로 C-Interface가 적용된 공유 라이브러리(`.so`)로 컴파일하는 전체 파이프라인을 제공합니다. 또한, 각 컴파일 단계별 메모리 사용량과 연산 빈도수를 추적하는 분석 도구와 시각화 대시보드를 포함합니다.

## 🛠 환경 세팅 (Prerequisites)
- **OS:** Ubuntu (WSL2 `5.15.167.4-microsoft-standard` 테스트 완료)
- **Python:** `3.12.3`
- **Compiler:** `CMake`, `Ninja`, `Clang`

### 1. 파이썬 가상환경 및 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate

# 요구 패키지 설치
pip install -r requirements.txt

cd ..
sudo apt install cmake ninja-build
git clone [https://github.com/llvm/llvm-project.git](https://github.com/llvm/llvm-project.git)
cd llvm-project
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release
ninja
cd ../../snn-mlir-compiler-profiler
