import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

# 웹 페이지 기본 설정
st.set_page_config(page_title="MLIR Profiling Dashboard", page_icon="⚙️", layout="wide")

st.title("SNN+DNN MLIR 컴파일러 프로파일링 대시보드")
st.markdown("추출된 `profile.json` 데이터를 기반으로 메모리 대역폭과 컴파일 하향(Lowering) 단계별 연산 팽창을 분석합니다.")

# 1. JSON 데이터 로드
@st.cache_data
def load_data():
    file_path = "outputs/profile.json"
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

data = load_data()

if data is None:
    st.error("'profile.json' 파일을 찾을 수 없습니다. 프로파일러 스크립트를 먼저 실행해주세요.")
    st.stop()

# 2. 탭(Tab) 구성
tab1, tab2 = st.tabs(["메모리 프로파일링 (Memory Profile)", "컴파일 레벨 연산 분석 (Operation Expansion)"])

# ==========================================
# 탭 1: 메모리 프로파일링
# ==========================================
with tab1:
    st.header("정적 메모리 및 동적 쓰기 대역폭 분석")
    mem_summary = data["memory_profiling"]["summary"]
    mem_details = data["memory_profiling"]["operations_bandwidth_details"]
    
    col1, col2 = st.columns(2)
    col1.metric("총 정적 메모리 (가중치/상수)", f"{mem_summary['total_static_memory_kb']} KB")
    col2.metric("분석된 동적 버퍼 할당 연산 수", f"{mem_summary['total_dynamic_ops_analyzed']} 개")
    
    if mem_details:
        st.subheader("연산별 동적 쓰기 대역폭 요구량 (Top Ops)")
        df_mem = pd.DataFrame(mem_details)
        df_mem = df_mem.sort_values(by="required_write_bandwidth_kb", ascending=False).head(10) # 상위 10개만
        
        # Plotly 막대 그래프
        fig_mem = px.bar(df_mem, x="required_write_bandwidth_kb", y="operation", 
                         orientation='h', text="required_write_bandwidth_kb",
                         labels={"required_write_bandwidth_kb": "요구 대역폭 (KB)", "operation": "Linalg 연산명"},
                         color="required_write_bandwidth_kb", color_continuous_scale="Reds")
        fig_mem.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_mem, use_container_width=True)

# ==========================================
# 탭 2: 컴파일 레벨 연산 분석
# ==========================================
with tab2:
    st.header("컴파일 하향(Lowering)에 따른 연산 증가")
    
    stats = data["compilation_statistics"]
    level_names = {
        "level_1_linalg_dialect": "Level 1: Linalg (고수준 연산)",
        "level_2_llvm_dialect": "Level 2: LLVM Dialect",
        "level_3_llvm_ir": "Level 3: LLVM IR (저수준 어셈블리)"
    }
    
    # 레벨 선택 라디오 버튼
    selected_key = st.radio("분석할 컴파일 레벨을 선택하세요:", list(level_names.keys()), format_func=lambda x: level_names[x], horizontal=True)
    
    selected_data = stats[selected_key]
    
    # 핵심 지표 표시
    st.metric("해당 레벨의 총 연산 개수 (Total Operations)", f"{selected_data['total_operations']:,} 개")
    st.info(f"**설명:** {selected_data['description']}")
    
    # 연산 종류별 빈도수 데이터프레임 변환
    ops_dict = selected_data["operation_details"]
    df_ops = pd.DataFrame(list(ops_dict.items()), columns=["Operation", "Count"])
    df_ops = df_ops.sort_values(by="Count", ascending=False).head(20) # 너무 많으면 상위 20개만 표시
    
    # Plotly 시각화
    fig_ops = px.bar(df_ops, x="Count", y="Operation", orientation='h', text="Count",
                     labels={"Count": "호출 횟수", "Operation": "명령어(Opcode)"},
                     color="Count", color_continuous_scale="Blues")
    fig_ops.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
    st.plotly_chart(fig_ops, use_container_width=True)
    
    # 컴파일러 인사이트 추가 표시
    if selected_key == "level_3_llvm_ir":
        st.success("**컴파일러 인사이트:** Linalg에서 약 200번 호출되었던 `generic` 연산이 메모리 주소 포인터 계산(`getelementptr`, `add`, `mul`)과 텐서 제어(`insertvalue`, `extractvalue`)로 인해 수천 개의 저수준 명령어로 팽창(Expansion)한 것을 확인할 수 있습니다.")
