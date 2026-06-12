# SNN-DNN MLIR Profiling Pipeline

이 프로젝트는 PyTorch 기반의 하이브리드 SNN-DNN 모델을 MLIR(Linalg, LLVM Dialect)을 거쳐 LLVM IR로 하향(Lowering)하고, 최종적으로 C-Interface가 적용된 공유 라이브러리(`.so`)로 컴파일하는 전체 파이프라인을 제공합니다. 또한, 각 컴파일 단계별 메모리 사용량과 연산 빈도수를 추적하는 분석 도구와 시각화 대시보드를 포함합니다.

## 환경 세팅 (Prerequisites)
- **OS:** Ubuntu (WSL2 `5.15.167.4-microsoft-standard` 테스트 완료)
- **Python:** `3.12.3`
- **Compiler:** `CMake`, `Ninja`, `Clang`

### 1. 파이썬 가상환경 및 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate

# 요구 패키지 설치
pip install -r requirements.txt
```

### 2. LLVM 툴체인 빌드
```bash
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
```

## 실행 가이드 (Quick Start)
### 단계 1: 전체 파이프라인 자동 실행
아래 쉘 스크립트를 실행하면 모델 추출부터 `.so` 컴파일까지 원클릭으로 진행됩니다.

```bash
chmod +x compile_pipeline.sh
./compile_pipeline.sh
```

### 단계 2: 연산 및 메모리 프로파일링
컴파일 된 산출물(`.mlir`, `.ll`)을 분석하여 `profile.json`을 생성합니다.

```bash
python3 tools/profiler.py
```
### 단계 3: Streamlit 시각화 대시보드 실행
추출된 JSON 데이터를 웹 브라우저에서 시각적으로 분석합니다.

```bash
streamlit run tools/dashboard.py
```
### 단계 4: 테스트벤치 실행
`ctyeps`를 이용해 `libmodel.so`를 로드하고 실행합니다.

```bash
python3 tests/run_model.py
```
