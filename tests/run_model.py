import ctypes
import numpy as np

# 1. 컴파일된 공유 라이브러리 로드
lib = ctypes.CDLL("outputs/libmodel.so")

# 2. MLIR MemRef와 1:1로 매칭되는 C 구조체 동적 생성 헬퍼
def make_memref_struct(rank, c_type=ctypes.c_float):
    class MemRefStruct(ctypes.Structure):
        if rank == 0:
            _fields_ = [
                ("allocated", ctypes.POINTER(c_type)),
                ("aligned", ctypes.POINTER(c_type)),
                ("offset", ctypes.c_int64),
            ]
        else:
            _fields_ = [
                ("allocated", ctypes.POINTER(c_type)),
                ("aligned", ctypes.POINTER(c_type)),
                ("offset", ctypes.c_int64),
                ("sizes", ctypes.c_int64 * rank),
                ("strides", ctypes.c_int64 * rank),
            ]
    return MemRefStruct

def numpy_to_memref(arr, rank, c_type=ctypes.c_float):
    StructType = make_memref_struct(rank, c_type)
    struct = StructType()
    c_ptr = arr.ctypes.data_as(ctypes.POINTER(c_type))
    
    struct.allocated = c_ptr
    struct.aligned = c_ptr
    struct.offset = 0
    if rank > 0:
        for i in range(rank):
            struct.sizes[i] = arr.shape[i]
            # strides는 byte 단위가 아니라 element 단위입니다.
            struct.strides[i] = arr.strides[i] // arr.itemsize
    return struct

print("데이터 버퍼 초기화 중...")

# 3. 모델이 요구하는 11개의 입력 데이터 준비
# (MLIR 파일의 %arg0 ~ %arg10 스펙에 정확히 맞춘 더미 텐서들)
a0 = numpy_to_memref(np.array(0.0, dtype=np.float32), 0, ctypes.c_float)
a1 = numpy_to_memref(np.array(0.0, dtype=np.float32), 0, ctypes.c_float)
a2 = numpy_to_memref(np.array(0, dtype=np.int64), 0, ctypes.c_int64) # 주의: i64 타입
a3 = numpy_to_memref(np.array(0.0, dtype=np.float32), 0, ctypes.c_float)
a4 = numpy_to_memref(np.empty((0,), dtype=np.float32), 1, ctypes.c_float)
a5 = numpy_to_memref(np.array(0.0, dtype=np.float32), 0, ctypes.c_float)
a6 = numpy_to_memref(np.array(0.0, dtype=np.float32), 0, ctypes.c_float)
a7 = numpy_to_memref(np.array(0, dtype=np.int64), 0, ctypes.c_int64) # 주의: i64 타입
a8 = numpy_to_memref(np.array(0.0, dtype=np.float32), 0, ctypes.c_float)
a9 = numpy_to_memref(np.empty((0,), dtype=np.float32), 1, ctypes.c_float)

# [핵심] 실제 네트워크 입력 (배치 4, 채널 1, 28x28 이미지)
img_data = np.random.randn(4, 1, 28, 28).astype(np.float32)
a10 = numpy_to_memref(img_data, 4, ctypes.c_float)

# 4. 결과값을 담을 빈 출력 버퍼 준비 (4 x 10 클래스 분류)
out_data = np.zeros((4, 10), dtype=np.float32)
out = numpy_to_memref(out_data, 2, ctypes.c_float)

# 5. C-Interface 함수 시그니처 정의 (반환값은 첫 번째 포인터로 넘어갑니다)
lib._mlir_ciface_main.argtypes = [
    ctypes.POINTER(type(out)),
    ctypes.POINTER(type(a0)), ctypes.POINTER(type(a1)), ctypes.POINTER(type(a2)),
    ctypes.POINTER(type(a3)), ctypes.POINTER(type(a4)), ctypes.POINTER(type(a5)),
    ctypes.POINTER(type(a6)), ctypes.POINTER(type(a7)), ctypes.POINTER(type(a8)),
    ctypes.POINTER(type(a9)), ctypes.POINTER(type(a10))
]

print("MLIR 공유 라이브러리(libmodel.so) 실행 중...")

# 6. 실행! (Segfault 없이 안전하게 포인터 전달)
lib._mlir_ciface_main(
    ctypes.byref(out),
    ctypes.byref(a0), ctypes.byref(a1), ctypes.byref(a2),
    ctypes.byref(a3), ctypes.byref(a4), ctypes.byref(a5),
    ctypes.byref(a6), ctypes.byref(a7), ctypes.byref(a8),
    ctypes.byref(a9), ctypes.byref(a10)
)

print("-" * 50)
print("[SUCCESS] 실행 완료! 추출된 최종 스파이크 출력 텐서 (4x10):")
print(out_data)
