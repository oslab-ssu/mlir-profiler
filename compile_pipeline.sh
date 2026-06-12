#!/bin/bash

# 색상 정의 (에러 메시지 강조용)
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# 에러 발생 시 메시지를 출력하고 스크립트를 종료하는 함수
check_status() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] $1 단계에서 실패했습니다.${NC}"
        exit 1
    else
        echo -e "${GREEN}[SUCCESS] $1 단계 완료.${NC}"
    fi
}

echo "=== 0. 파이썬 모델 실행 및 MLIR 추출 ==="
python3 dnnsnn.py
check_status "python3 dnnsnn.py 실행"

echo -e "\n=== 1. 텐서를 메모리 버퍼로 변환 (Bufferization) ==="
../llvm-project/build/bin/mlir-opt snn_dnn_linalg.mlir \
  -empty-tensor-to-alloc-tensor \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -o snn_dnn_bufferized.mlir
check_status "1단계: Bufferization (mlir-opt)"

sed -i 's/-> memref<4x10xf32> {/-> memref<4x10xf32> attributes {llvm.emit_c_interface} {/' snn_dnn_bufferized.mlir

echo -e "\n=== 2. Linalg 및 제어 흐름을 LLVM IR로 하향 (-cpu-lower 과정) ==="
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

check_status "2단계: LLVM Dialect 변환 (mlir-opt)"

echo -e "\n=== 3. MLIR을 LLVM IR(.ll)로 번역 ==="
../llvm-project/build/bin/mlir-translate -mlir-to-llvmir snn_dnn_llvm_dialect.mlir -o snn_dnn_llvm_ir.ll
check_status "3단계: LLVM IR 번역 (mlir-translate)"

echo -e "\n=== 4. Clang을 사용해 최적화(-O3) 및 공유 라이브러리(.so) 컴파일 ==="
../llvm-project/build/bin/clang -O3 -g -shared -fPIC snn_dnn_llvm_ir.ll \
    -L../llvm-project/build/lib \
    -lmlir_c_runner_utils \
    -lmlir_runner_utils \
    -Wl,-rpath,$(pwd)/../llvm-project/build/lib \
    -o libmodel.so
check_status "4단계: 최종 .so 컴파일 (clang)"

echo -e "\n${GREEN} 모든 컴파일 단계가 성공적으로 완료되었습니다! (libmodel.so 생성 완료)${NC}"
