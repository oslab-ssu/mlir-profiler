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
---
## 특정 구간 연산 실행 시간 측정
MLIR 패스(Pass) 전환 과정 중 특정 연산 구간의 실제 실행 시간을 정밀하게 측정하기 위해, C 기반의 타이머를 설계하고 MLIR 계층에 주입하여 공유 라이브러리로 빌드하는 방법을 제공합니다.

### 1. 타이머 코드 작성 (`timer.c`)
가장 먼저 Monotonic Clock을 활용하여 나노초 단위까지 측정 가능한 타이머 C 소스코드를 작성합니다.

### 2. 코드 삽입 (`snn_dnn_bufferized.mlir`)
#### 2.1 최상단 외부 함수 선언
module 기호 바로 아래에 C에서 작성한 `start_timer`와 `stop_timer` 함수를 외부 함수(`private`)로 선언합니다.

#### 2.2 측정 대상 커널 범위 래핑 (Wrapping)
측정하고자 하는 Linalg 연산 영역 앞뒤로 `func.call`을 이용해 타이머 함수를 배치합니다. `stop_timer` 호출 시에는 구간 구분을 위한 `i32` ID 상수를 전달합니다.

```mlir
// ... 이전 연산 생략 ...
    linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_28, %arg0 : memref<4x64xf32>, memref<f32, strided<[], offset: ?>>) outs(%alloc_6 : memref<4x64xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %10 = arith.subf %in, %in_142 : f32
      linalg.yield %10 : f32
    }
    
    // ─── [Section 1] 타이머 시작 ───
    func.call @start_timer() : () -> ()
    
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_6 : memref<4x64xf32>) outs(%alloc_8 : memref<4x64xi1>) {
    ^bb0(%in: f32, %out: i1):
      %10 = arith.cmpf ogt, %in, %cst : f32
      linalg.yield %10 : i1
    }
    // ... 중략 (계층 연산) ...
    linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_73, %2 : memref<4x10xf32>, memref<10xf32>) outs(%alloc_74 : memref<4x10xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %10 = arith.addf %in, %in_142 : f32
      linalg.yield %10 : f32
    }
    
    // ─── [Section 1] 타이머 종료 ───
    %section_id_1 = arith.constant 1 : i32
    func.call @stop_timer(%section_id_1) : (i32) -> ()
    
    // ─── [Section 2] 타이머 시작 ───
    func.call @start_timer() : () -> ()
    
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_49 : memref<4x32xf32>) outs(%alloc_49 : memref<4x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %10 = arith.cmpf ugt, %in, %cst : f32
      %11 = arith.select %10, %in, %cst : f32
      linalg.yield %11 : f32
    }
    // ... 중략 ...
    linalg.generic {indexing_maps = [#map1, #map3, #map1], iterator_types = ["parallel", "parallel"]} ins(%alloc_102, %arg5 : memref<4x10xf32>, memref<f32, strided<[], offset: ?>>) outs(%alloc_53 : memref<4x10xf32>) {
    ^bb0(%in: f32, %in_142: f32, %out: f32):
      %10 = arith.subf %in, %in_142 : f32
      linalg.yield %10 : f32
    }
    
    // ─── [Section 2] 타이머 종료 ───
    %section_id_2 = arith.constant 2 : i32
    func.call @stop_timer(%section_id_2) : (i32) -> ()
```

### 3. 컴파일 및 공유 라이브러리 빌드
#### A. Bufferized MLIR → LLVM Dialect 하향
```bash
../llvm-project/build/bin/mlir-opt snn_dnn_bufferized.mlir \
      --canonicalize \
      -convert-linalg-to-loops \
      -lower-affine \
      -convert-scf-to-cf \
      -convert-math-to-llvm \
      -convert-arith-to-llvm \
      -expand-strided-metadata \
      -finalize-memref-to-llvm \
      -convert-cf-to-llvm \
      -convert-func-to-llvm \
      -reconcile-unrealized-casts \
      --mlir-timing \
      --mlir-output-format=text \
      -o snn_dnn_llvm_dialect.mlir
```
#### B. LLVM Dialect → LLVM IR 변환
```bash
../llvm-project/build/bin/mlir-translate -mlir-to-llvmir snn_dnn_llvm_dialect.mlir -o snn_dnn_llvm_ir.ll
```
#### C. Clang 컴파일러를 이용한 타이머 링킹 및 공유 라이브러리(`.so`) 생성
MLIR 러너 유틸리티 라이브러리(`mlir_c_runner_utils`)를 링크하여 타이머와 함께 패키징합니다.
```bash
../llvm-project/build/bin/clang -O3 -g -shared -fPIC snn_dnn_llvm_ir.ll timer.c \             
    -L../llvm-project/build/lib \
    -lmlir_c_runner_utils \
    -lmlir_runner_utils \
    -Wl,-rpath,$(pwd)/../llvm-project/build/lib \
    -o libmodel.so
```

### 4. 파이썬 C-Interface 연동 검증 (`run_model.py`)
```bash
python3 run_model.py
```
#### 실행 로그 결과 예시
```
데이터 버퍼 초기화 중...
MLIR 공유 라이브러리(libmodel.so) 실행 중...
[Profiler] Section 1 Execution Time: 0.000053 seconds
[Profiler] Section 2 Execution Time: 0.000007 seconds
--------------------------------------------------
[SUCCESS] 실행 완료! 추출된 최종 스파이크 출력 텐서 (4x10):
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
```
