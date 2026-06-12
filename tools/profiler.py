import re
import json
import os
from collections import Counter

# ==========================================
# 1. 메모리 프로파일링 (정적 메모리 및 버퍼 대역폭)
# ==========================================
def calc_bytes(shape_str):
    parts = shape_str.split('x')
    dtype = parts[-1]
    dims = [d for d in parts[:-1] if d.isdigit() or d == '?']
    
    bytes_per_elem = 4 # f32 기본값
    if 'f16' in dtype: bytes_per_elem = 2
    elif 'i1' in dtype or 'i8' in dtype: bytes_per_elem = 1
    elif 'f64' in dtype or 'i64' in dtype: bytes_per_elem = 8
    
    total_elems = 1
    for d in dims:
        if d != '?':
            total_elems *= int(d)
    return total_elems * bytes_per_elem

def profile_memory_to_dict(mlir_file):
    static_mem_bytes = 0
    allocs = {}  
    dynamic_ops = []

    if not os.path.exists(mlir_file):
        return {"error": f"{mlir_file} not found."}

    with open(mlir_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if 'memref.global' in line:
            match = re.search(r'memref<([^>]+)>', line)
            if match:
                static_mem_bytes += calc_bytes(match.group(1))

    for line in lines:
        alloc_match = re.search(r'(%[a-zA-Z0-9_]+)\s*=\s*memref\.alloc.*memref<([^>]+)>', line)
        if alloc_match:
            alloc_id = alloc_match.group(1)
            size_bytes = calc_bytes(alloc_match.group(2))
            allocs[alloc_id] = size_bytes
            continue
            
        linalg_match = re.search(r'linalg\.([a-zA-Z0-9_]+).*outs\(([^)]+)\)', line)
        if linalg_match:
            op_name = linalg_match.group(1)
            outs_args = linalg_match.group(2).split(',')
            
            for arg in outs_args:
                arg = arg.strip().split(':')[0].strip() 
                if arg in allocs:
                    req_kb = allocs[arg] / 1024
                    dynamic_ops.append({
                        "operation": f"linalg.{op_name}",
                        "output_buffer_id": arg,
                        "required_write_bandwidth_kb": round(req_kb, 2)
                    })

    return {
        "summary": {
            "total_static_memory_kb": round(static_mem_bytes / 1024, 2),
            "total_dynamic_ops_analyzed": len(dynamic_ops)
        },
        "operations_bandwidth_details": dynamic_ops
    }

# ==========================================
# 2. 연산자 빈도수 프로파일링
# ==========================================
def count_regex_ops(filepath, pattern):
    """정규식을 사용하여 특정 다이얼렉트의 연산을 추출하고 카운트합니다."""
    op_counter = Counter()
    if not os.path.exists(filepath):
        return 0, {}
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        matches = re.findall(pattern, content)
        op_counter.update(matches)
        
    sorted_ops = {k: v for k, v in sorted(op_counter.items(), key=lambda item: item[1], reverse=True)}
    return sum(op_counter.values()), sorted_ops

def count_llvm_ir_ops(filepath):
    """awk 스크립트의 로직을 파이썬으로 완벽하게 포팅하여 LLVM IR의 실제 연산만 카운트합니다."""
    op_counter = Counter()
    if not os.path.exists(filepath):
        return 0, {}
        
    ignore_ops = {'br', 'phi', 'icmp', 'private', 'declare', 'attributes', 'define', 'ret', 'source_filename', 'target', 'dso_local', 'module'}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # 주석 및 메타데이터, 분기 라벨 무시
            if line.startswith(';') or line.startswith('!') or line.startswith('"'): continue
            if re.match(r'^[0-9a-zA-Z_.]+:', line): continue
            
            parts = line.split()
            if not parts: continue
            
            op = None
            # 1. SSA 할당문인지 확인 (%146 = add ...)
            if parts[0].startswith('%') and len(parts) >= 3 and parts[1] == '=':
                op = parts[2]
            # 2. 할당 없는 실행 명령어인지 확인 (store, call 등)
            elif parts[0][0].isalpha():
                op = parts[0]
            
            # 필터링 및 카운트
            if op and op not in ignore_ops:
                op_counter[op] += 1
                
    sorted_ops = {k: v for k, v in sorted(op_counter.items(), key=lambda item: item[1], reverse=True)}
    return sum(op_counter.values()), sorted_ops

# ==========================================
# 3. 데이터 통합 및 JSON 출력
# ==========================================
def generate_comprehensive_profile():
    print("프로파일링 데이터를 추출하고 JSON으로 병합 중입니다...")
    
    # 1. 메모리 프로파일링 추출
    memory_data = profile_memory_to_dict("snn_dnn_bufferized.mlir")
    
    # 2. 단계별 연산 빈도수 추출
    linalg_total, linalg_ops = count_regex_ops("outputs/snn_dnn_bufferized.mlir", r'linalg\.[a-zA-Z0-9_]+')
    llvm_dialect_total, llvm_dialect_ops = count_regex_ops("outputs/snn_dnn_llvm_dialect.mlir", r'llvm\.[a-zA-Z0-9_]+')
    llvm_ir_total, llvm_ir_ops = count_llvm_ir_ops("outputs/snn_dnn_llvm_ir.ll")
    
    # 3. 최종 JSON 구조 조립
    final_json = {
        "memory_profiling": memory_data,
        "compilation_statistics": {
            "level_1_linalg_dialect": {
                "description": "고수준 수학 연산 및 버퍼 할당 레벨",
                "total_operations": linalg_total,
                "operation_details": linalg_ops
            },
            "level_2_llvm_dialect": {
                "description": "LLVM IR과 1:1 매칭되는 하위 다이얼렉트 레벨",
                "total_operations": llvm_dialect_total,
                "operation_details": llvm_dialect_ops
            },
            "level_3_llvm_ir": {
                "description": "어셈블리 직전의 순수 수학/메모리 연산 레벨",
                "total_operations": llvm_ir_total,
                "operation_details": llvm_ir_ops
            }
        }
    }
    
    # 4. 파일로 저장 및 터미널 출력
    output_filename = "outputs/profile.json"
    json_string = json.dumps(final_json, indent=4, ensure_ascii=False)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(json_string)
        
    print(json_string)
    print(f"\n[SUCCESS] 모든 분석이 완료되어 '{output_filename}'에 저장되었습니다.")

if __name__ == "__main__":
    generate_comprehensive_profile()
