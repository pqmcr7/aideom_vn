from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.data_loader import load_data, ensure_standard_files
from src import optimization as m
from src.charts import (
    metric_cards_html,
    line_chart,
    bar_chart,
    heatmap_from_matrix,
    sensitivity_heatmap,
    pareto_scatter,
    parallel_coordinates,
    radar_chart,
    labor_sankey,
)
from src.reporting import build_pdf_summary, build_markdown_report

# -----------------------------------------------------------------------------
# App config and style
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AIDEOM-VN Dashboard",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(metric_cards_html(), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cache wrappers
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_data():
    ensure_standard_files()
    return load_data()

@st.cache_data(show_spinner="Đang chạy Pareto approximation...")
def cached_pareto(seed: int, n_samples: int, weights_tuple: tuple[float, float, float, float]):
    return m.pareto_digital_budget(seed=seed, n_samples=n_samples, weights=weights_tuple)

@st.cache_data(show_spinner="Đang tối ưu động 2026–2035...")
def cached_dynamic(discount: float, seed: int, samples: int):
    return m.optimize_dynamic(discount=discount, seed=seed, n_samples=samples)

@st.cache_data(show_spinner="Đang huấn luyện Q-learning...")
def cached_qlearning(episodes: int, lr: float, gamma: float, seed: int, weights_tuple: tuple[float, float, float, float]):
    return m.train_q_learning(episodes=episodes, alpha=lr, discount=gamma, seed=seed, reward_weights=weights_tuple)

@st.cache_data(show_spinner="Đang so sánh kịch bản...")
def cached_scenarios(discount: float):
    return m.compare_scenarios(discount=discount)


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("# ⚙️ AIDEOM-VN")
    st.caption("Tất cả tham số cấu hình được đặt ở thanh bên để màn hình chính chỉ hiển thị kết quả.")

    if st.button("🔄 Reset cache / chạy lại mô hình", use_container_width=True):
        st.cache_data.clear()
        st.success("Đã xóa cache. Mô hình sẽ chạy lại ở lần render tiếp theo.")
        st.rerun()

    st.divider()
    with st.expander("📌 Bài 1 – Cobb-Douglas", expanded=False):
        alpha = st.slider("α – vốn K", 0.10, 0.60, 0.33, 0.01)
        beta = st.slider("β – lao động L", 0.10, 0.70, 0.42, 0.01)
        gamma_cd = st.slider("γ – số hóa D", 0.00, 0.30, 0.10, 0.01)
        delta = st.slider("δ – AI", 0.00, 0.30, 0.08, 0.01)
        theta = st.slider("θ – nhân lực H", 0.00, 0.30, 0.07, 0.01)
        st.caption(f"Tổng hệ số = {alpha + beta + gamma_cd + delta + theta:.2f}")
        d2030 = st.slider("D mục tiêu 2030 (%)", 20.0, 40.0, 30.0, 0.5)
        ai2030 = st.slider("AI 2030 (nghìn DN)", 80.0, 150.0, 100.0, 1.0)
        h2030 = st.slider("H 2030 (%)", 25.0, 45.0, 35.0, 0.5)

    with st.expander("💰 Bài 2/5/10 – Ngân sách", expanded=False):
        budget_b2 = st.slider("Bài 2: Tổng ngân sách LP", 80.0, 160.0, 100.0, 5.0)
        x3_min = st.slider("Bài 2: Sàn nhân lực số x3", 20.0, 50.0, 20.0, 1.0)
        budget_b5 = st.slider("Bài 5: Ngân sách MIP", 60000, 110000, 80000, 5000)
        budget_b10 = st.slider("Bài 10: Ngân sách SP", 60000, 120000, 80000, 5000)
        reserve_b10 = st.slider("Bài 10: Quỹ dự phòng recourse", 5000, 30000, 15000, 1000)
        scenario_scale = st.slider("Bài 10: Mức độ bất định", 0.0, 2.0, 1.0, 0.1)

    with st.expander("⚖️ Trọng số Bài 3/6/7/11", expanded=False):
        w_growth = st.slider("Bài 3: tăng trưởng", 0.00, 0.40, 0.15, 0.01)
        w_prod = st.slider("Bài 3: năng suất/GDP share", 0.00, 0.40, 0.15, 0.01)
        w_spill = st.slider("Bài 3: lan tỏa", 0.00, 0.40, 0.20, 0.01)
        w_export = st.slider("Bài 3: xuất khẩu", 0.00, 0.40, 0.15, 0.01)
        w_emp = st.slider("Bài 3: việc làm", 0.00, 0.40, 0.10, 0.01)
        w_ai = st.slider("Bài 3/6: AI readiness", 0.00, 0.50, 0.20, 0.01)
        w_risk = st.slider("Bài 3: phạt rủi ro", 0.00, 0.40, 0.15, 0.01)
        st.markdown("**Bài 7/11: ưu tiên chính sách**")
        wg = st.slider("Tăng trưởng", 0.00, 1.00, 0.40, 0.05)
        wi = st.slider("Bao trùm / Digital", 0.00, 1.00, 0.25, 0.05)
        we = st.slider("Môi trường / việc làm", 0.00, 1.00, 0.20, 0.05)
        ws = st.slider("An ninh / phát thải", 0.00, 1.00, 0.15, 0.05)

    with st.expander("🧭 Bài 4 – Công bằng vùng", expanded=False):
        lam = st.slider("λ công bằng", 0.40, 0.95, 0.70, 0.01)
        soft_fairness = st.checkbox("Cho phép soft fairness slack", value=True)

    with st.expander("⏳ Bài 7/8/11 – Hiệu năng", expanded=False):
        pareto_samples = st.slider("Số mẫu Pareto", 300, 3000, 1200, 100)
        dynamic_samples = st.slider("Số mẫu tối ưu động", 200, 2500, 700, 100)
        q_episodes = st.slider("Q-learning episodes", 300, 10000, 2500, 100)
        q_lr = st.slider("Q-learning α", 0.01, 0.50, 0.10, 0.01)
        q_gamma = st.slider("Q-learning γ discount", 0.50, 0.99, 0.95, 0.01)
        seed = st.number_input("Seed", min_value=1, max_value=9999, value=42)

    with st.expander("📉 Bài 8 – Chiết khấu", expanded=False):
        discount = st.slider("Tỷ lệ chiết khấu ρ", 0.85, 0.995, 0.97, 0.005)

    with st.expander("🧑‍🏭 Bài 9/14 – Lao động", expanded=False):
        ai_budget_lab = st.slider("Ngân sách AI theo ngành", 4000, 25000, 12000, 1000)
        h_budget_lab = st.slider("Ngân sách đào tạo lại", 4000, 25000, 10000, 1000)
        enforce_labor_cap = st.checkbox("Bài 9: thêm ràng buộc không ngành nào mất >5% lao động", value=False)
        exclude_mining = st.checkbox("Tab 14: phân tích 9 ngành, loại Khai khoáng", value=True)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

macro, sectors, regions = cached_data()

# Prepare frequently used outputs
weights_b3 = {"growth": w_growth, "productivity": w_prod, "spillover": w_spill, "export": w_export, "employment": w_emp, "ai": w_ai, "risk": w_risk}
policy_weights = np.array([wg, wi, we, ws], dtype=float)
if np.isclose(policy_weights.sum(), 0):
    policy_weights = np.array([0.40, 0.25, 0.20, 0.15])
policy_weights = tuple((policy_weights / policy_weights.sum()).round(5))



def section_header(title: str, badge: str = "Interactive"):
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{title}</h2>
            <span class="chip">✨ {badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

if "ui_welcome_shown" not in st.session_state:
    st.toast("Dashboard AIDEOM-VN đã sẵn sàng 🚀", icon="✅")
    st.session_state["ui_welcome_shown"] = True

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero">
      <h1>🇻🇳 AIDEOM-VN Dashboard</h1>
      <p class="subtitle">
        Mô hình ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI — kết hợp dữ liệu vĩ mô, ngành, vùng
        với LP, MIP, TOPSIS, Pareto, tối ưu động, Stochastic Programming và Q-learning.
      </p>
      <div class="chip-row">
        <span class="chip">📊 14 tab phân tích</span>
        <span class="chip">🧠 12 mô hình chính</span>
        <span class="chip">⚡ Cache mô hình nặng</span>
        <span class="chip">☁️ Sẵn sàng deploy Streamlit Cloud</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("GDP 2025", f"{macro['GDP_trillion_VND'].iloc[-1]:,.1f}", "nghìn tỷ VND")
with kpi2:
    st.metric("Kinh tế số/GDP 2025", f"{macro['digital_economy_share_GDP_pct'].iloc[-1]:.1f}%")
with kpi3:
    st.metric("Số ngành", len(sectors))
with kpi4:
    st.metric("Số vùng", len(regions))

with st.expander("🧭 Hướng dẫn nhanh sử dụng dashboard", expanded=False):
    st.markdown(
        """
        - Chọn tab theo từng bài để xem **kết quả số**, **biểu đồ** và **phân tích chính sách**.
        - Toàn bộ tham số như ngân sách, trọng số, λ công bằng, số episode Q-learning được đặt ở **sidebar**.
        - Bấm **Reset cache / chạy lại mô hình** nếu muốn buộc các mô hình nặng chạy lại từ đầu.
        - Cuối trang có nút **tải báo cáo Markdown/PDF** để nộp kèm bài.
        """
    )

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tabs = st.tabs([
    "📊 1. Tổng quan dữ liệu",
    "🏭 2. Bài 1 – Hàm sản xuất",
    "💰 3. Bài 2 – LP ngân sách",
    "🏆 4. Bài 3 – Priority ngành",
    "🗺️ 5. Bài 4 – LP vùng",
    "📦 6. Bài 5 – MIP dự án",
    "🧭 7. Bài 6 – TOPSIS",
    "🌈 8. Bài 7 – Pareto",
    "⏳ 9. Bài 8 – Tối ưu động",
    "🧑‍🏭 10. Bài 9 – AI lao động",
    "🎲 11. Bài 10 – Stochastic",
    "🤖 12. Bài 11 – Q-learning",
    "📡 13. Bài 12 – S1–S5",
    "🔎 14. Chính sách ngành",
])

# Tab 1
with tabs[0]:
    section_header("📊 Tổng quan dữ liệu đầu vào")
    st.markdown("Dữ liệu được đọc từ 3 CSV. Nếu thiếu file, app tự tạo dữ liệu fallback từ đề bài.")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(line_chart(macro, "year", ["GDP_trillion_VND", "exports_billion_USD", "FDI_disbursed_billion_USD"], "Macro indicators 2020–2025"), use_container_width=True)
    with c2:
        st.plotly_chart(bar_chart(sectors.sort_values("gdp_share_2024_pct", ascending=False), "sector_name_vi", "gdp_share_2024_pct", "Tỷ trọng GDP theo ngành", orientation="v"), use_container_width=True)
    with st.expander("📄 Bảng macro"):
        st.dataframe(macro, use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        with st.expander("🏭 Bảng ngành"):
            st.dataframe(sectors, use_container_width=True)
    with c4:
        with st.expander("🗺️ Bảng vùng"):
            st.dataframe(regions, use_container_width=True)

# Tab 2 / Bài 1
with tabs[1]:
    section_header("🏭 Bài 1 – Hàm sản xuất Cobb-Douglas mở rộng")
    b1 = m.cobb_douglas_analysis(macro, alpha, beta, gamma_cd, delta, theta, d2030, ai2030, h2030)
    c1, c2, c3 = st.columns(3)
    c1.metric("TFP A_mean", f"{b1['A_mean']:.4f}")
    c2.metric("MAPE", f"{b1['mape']:.2f}%")
    c3.metric("GDP dự báo 2030", f"{b1['forecast']['GDP_forecast'].iloc[-1]:,.1f}")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(line_chart(b1["summary"], "year", ["TFP_A"], "TFP A_t theo năm"), use_container_width=True)
    with col2:
        st.plotly_chart(line_chart(b1["forecast"], "year", ["GDP_forecast"], "Mô phỏng GDP đến 2030"), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(bar_chart(b1["growth"], "Thành phần", "Đóng góp bình quân (% điểm/năm)", "Phân rã tăng trưởng"), use_container_width=True)
    with col4:
        st.markdown("### 🧠 Phân tích")
        top_comp = b1["growth"].sort_values("Đóng góp bình quân (% điểm/năm)", ascending=False).iloc[0]
        st.markdown(f"- Thành phần đóng góp lớn nhất là **{top_comp['Thành phần']}**.")
        st.markdown("- Nếu D đạt 30%, AI đạt 100 nghìn DN và H đạt 35%, GDP 2030 tăng mạnh trong mô phỏng Cobb-Douglas.")
        st.markdown("- TFP tăng ổn định phản ánh chất lượng tăng trưởng tốt hơn, nhưng phụ thuộc giả định hệ số co giãn.")
    with st.expander("📋 Bảng kết quả chi tiết"):
        st.dataframe(b1["summary"], use_container_width=True)
        st.dataframe(b1["growth"], use_container_width=True)
        st.dataframe(b1["forecast"], use_container_width=True)

# Tab 3 / Bài 2
with tabs[2]:
    section_header("💰 Bài 2 – LP phân bổ ngân sách 4 hạng mục")
    b2 = m.solve_budget_lp(budget_b2, x3_min)
    c1, c2, c3 = st.columns(3)
    c1.metric("Trạng thái", b2["status"])
    c2.metric("Z* GDP gain", f"{b2['z']:,.2f}")
    c3.metric("Shadow price ngân sách", f"{b2['shadow_budget']:,.3f}")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(bar_chart(b2["allocation"], "Hạng mục", "Phân bổ (nghìn tỷ VND)", "Phân bổ tối ưu"), use_container_width=True)
    with col2:
        st.plotly_chart(line_chart(b2["sensitivity"], "Ngân sách", "Z*", "Độ nhạy Z*(B)"), use_container_width=True)
    with st.expander("📋 Bảng LP và diễn giải shadow price"):
        st.dataframe(b2["allocation"], use_container_width=True)
        st.dataframe(b2["sensitivity"], use_container_width=True)
        st.markdown(f"Shadow price ngân sách xấp xỉ **{b2['shadow_budget']:.3f}** nghĩa là tăng thêm 1 nghìn tỷ VND ngân sách làm Z* tăng khoảng giá trị này trong miền còn hiệu lực.")

# Tab 4 / Bài 3
with tabs[3]:
    section_header("🏆 Bài 3 – Chỉ số ưu tiên ngành Priorityᵢ")
    b3 = m.priority_index(sectors, weights_b3)
    st.success("Top-3 ưu tiên: " + ", ".join(b3["top3"]))
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.plotly_chart(bar_chart(b3["ranking"], "sector_name_vi", "Priority", "Xếp hạng Priority ngành"), use_container_width=True)
    with col2:
        st.plotly_chart(sensitivity_heatmap(b3["sensitivity"], "Độ nhạy thứ hạng khi thay đổi trọng số AI"), use_container_width=True)
    with st.expander("📋 Bảng chuẩn hóa và ranking"):
        st.dataframe(b3["ranking"], use_container_width=True)
        st.dataframe(b3["normalized"], use_container_width=True)

# Tab 5 / Bài 4
with tabs[4]:
    section_header("🗺️ Bài 4 – LP vùng × hạng mục có ràng buộc công bằng")
    b4 = m.solve_regional_lp(lam=lam, soft=soft_fairness)
    c1, c2, c3 = st.columns(3)
    c1.metric("Trạng thái", b4["status"])
    c2.metric("Z* có công bằng", f"{b4['z']:,.1f}")
    c3.metric("Chi phí công bằng", f"{b4['fairness_cost']:,.1f}")
    col1, col2 = st.columns([1.15, .85])
    with col1:
        st.plotly_chart(heatmap_from_matrix(b4["allocation"], "Heatmap phân bổ ngân sách tối ưu"), use_container_width=True)
    with col2:
        if len(b4["region_sum"]):
            st.plotly_chart(bar_chart(b4["region_sum"], "Vùng", "Tổng ngân sách", "Tổng ngân sách theo vùng"), use_container_width=True)
    with st.expander("📋 Ma trận phân bổ, slack công bằng và so sánh không công bằng"):
        st.dataframe(b4["allocation"], use_container_width=True)
        st.dataframe(b4["slack"].reset_index().rename(columns={"index": "Vùng"}), use_container_width=True)
        st.markdown(f"Mô hình không có C5 đạt Z ≈ **{b4['nofair_z']:,.1f}**, chênh lệch phản ánh chi phí kinh tế của ràng buộc công bằng vùng.")

# Tab 6 / Bài 5
with tabs[5]:
    section_header("📦 Bài 5 – MIP lựa chọn 15 dự án chuyển đổi số")
    risk_adjust = st.checkbox("Dùng lợi ích kỳ vọng theo xác suất hoàn thành", value=False, key="riskadj")
    force_redundancy = st.checkbox("Bắt buộc chọn cả P1 và P2", value=False, key="p1p2")
    b5 = m.solve_project_mip(budget_b5, force_p1_p2=force_redundancy, expected_value=risk_adjust)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trạng thái", b5["status"])
    c2.metric("Tổng chi phí", f"{b5['summary']['cost']:,.0f}")
    c3.metric("Tổng lợi ích", f"{b5['summary']['Z']:,.0f}")
    c4.metric("NPV biên", f"{b5['summary']['ratio']:.2f}")
    if len(b5["selected"]):
        col1, col2 = st.columns([.9, 1.1])
        with col1:
            st.plotly_chart(bar_chart(b5["selected"], "name", "cost", "Chi phí các dự án được chọn", color="field"), use_container_width=True)
        with col2:
            st.dataframe(b5["selected"], use_container_width=True)
    with st.expander("📋 Danh mục 15 dự án gốc"):
        st.dataframe(m.PROJECTS, use_container_width=True)

# Tab 7 / Bài 6
with tabs[6]:
    section_header("🧭 Bài 6 – TOPSIS xếp hạng vùng ưu tiên AI")
    base_w6 = np.array([0.10, 0.10, 0.15, max(w_ai, 0.01), 0.15, 0.15, 0.05, 0.10], dtype=float)
    b6 = m.topsis_regions(regions, base_w6, entropy=False)
    b6e = m.topsis_regions(regions, entropy=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(bar_chart(b6["ranking"], "region_name_vi", "TOPSIS_score", "TOPSIS với trọng số chuyên gia"), use_container_width=True)
        st.dataframe(b6["ranking"], use_container_width=True)
    with col2:
        st.plotly_chart(bar_chart(b6e["ranking"], "region_name_vi", "TOPSIS_score", "TOPSIS với Entropy weights"), use_container_width=True)
        st.dataframe(b6e["weights"], use_container_width=True)
    st.markdown("🧠 **Gợi ý chính sách:** vùng top-3 là ứng viên triển khai trung tâm AI/sandbox dữ liệu, nhưng cần xét thêm tiêu chí địa–chính trị và công bằng vùng.")

# Tab 8 / Bài 7
with tabs[7]:
    section_header("🌈 Bài 7 – Tối ưu đa mục tiêu Pareto")
    b7 = cached_pareto(int(seed), pareto_samples, policy_weights)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số nghiệm Pareto", len(b7["pareto"]))
    c2.metric("GDP best compromise", f"{b7['opportunity']['GDP_gain_best']:,.1f}")
    c3.metric("GDP max-growth", f"{b7['opportunity']['GDP_gain_max']:,.1f}")
    col1, col2 = st.columns([1.1, .9])
    with col1:
        st.plotly_chart(pareto_scatter(b7["pareto"]), use_container_width=True)
    with col2:
        st.plotly_chart(parallel_coordinates(b7["pareto"], "Parallel coordinates trên tập Pareto"), use_container_width=True)
    with st.expander("📋 Nghiệm thỏa hiệp và cơ hội đánh đổi"):
        st.dataframe(b7["best_allocation"], use_container_width=True)
        st.json(b7["opportunity"])
        st.dataframe(b7["pareto"].sort_values("CompromiseScore", ascending=False).head(20), use_container_width=True)

# Tab 9 / Bài 8
with tabs[8]:
    section_header("⏳ Bài 8 – Tối ưu động phân bổ liên thời gian 2026–2035")
    b8 = cached_dynamic(discount, int(seed), dynamic_samples)
    path8 = b8["path"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Welfare", f"{b8['welfare']:.2f}")
    c2.metric("K share", f"{b8['shares'][0]:.1%}")
    c3.metric("D/AI share", f"{b8['shares'][1]:.1%} / {b8['shares'][2]:.1%}")
    c4.metric("H share", f"{b8['shares'][3]:.1%}")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(line_chart(path8, "year", ["GDP", "Consumption"], "GDP và tiêu dùng theo thời gian"), use_container_width=True)
    with col2:
        st.plotly_chart(line_chart(path8, "year", ["D", "AI", "H"], "Tích lũy số hóa, AI, nhân lực"), use_container_width=True)
    with st.expander("📋 Bảng đường đi tối ưu động"):
        st.dataframe(path8, use_container_width=True)

# Tab 10 / Bài 9
with tabs[9]:
    st.subheader("🧑‍🏭 Bài 9 – Tác động AI tới lao động và NetJob")
    b9 = m.labor_ai_impact(sectors, ai_budget_lab, h_budget_lab, exclude_mining=False, enforce_5pct=enforce_labor_cap)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trạng thái LP", b9["status"])
    c2.metric("Tổng NetJob", f"{b9['table']['NetJob_jobs'].sum():,.0f} việc")
    c3.metric("Ngành rủi ro cao nhất", b9["table"].sort_values("Displaced_pct_labor", ascending=False).iloc[0]["sector_name_vi"])
    c4.metric("Ngưỡng x_H ngành 2", f"{b9['threshold'].iloc[0]['x_H tối thiểu cần dùng']:,.0f}")

    col1, col2 = st.columns([1.05, .95])
    with col1:
        st.plotly_chart(bar_chart(b9["table"], "sector_name_vi", "NetJob_jobs", "NetJob ròng theo ngành (việc)"), use_container_width=True)
    with col2:
        st.plotly_chart(bar_chart(b9["table"], "sector_name_vi", "Displaced_pct_labor", "Tỷ lệ lao động bị dịch chuyển (%)"), use_container_width=True)

    col3, col4 = st.columns([1.2, .8])
    with col3:
        st.plotly_chart(labor_sankey(b9["sankey"]), use_container_width=True)
    with col4:
        st.markdown("### 🧠 Phân tích chính sách")
        st.markdown("- Mô hình đã bổ sung đúng cấu trúc **NewJob + UpgradeJob - DisplacedJob**.")
        st.markdown("- Ràng buộc **DisplacedJobᵢ ≤ RetrainingCapacityᵢ** thể hiện nguyên tắc tốc độ tự động hóa không vượt quá năng lực đào tạo lại.")
        st.markdown("- Kiểm tra mở rộng 5% giúp đánh giá tính an sinh xã hội của phương án AI.")

    with st.expander("📋 Bảng NetJob, threshold ngành 2, Sankey data và kiểm tra ràng buộc 5%"):
        st.markdown("**Kết quả LP theo 8 ngành trong đề bài**")
        st.dataframe(b9["table"], use_container_width=True)
        st.markdown("**Câu 9.4.2 — Ngưỡng đầu tư đào tạo tối thiểu ở ngành 2**")
        st.dataframe(b9["threshold"], use_container_width=True)
        st.markdown("**Câu 9.4.3 — Dữ liệu Sankey nhóm dễ bị tổn thương ngành 1, 3, 4**")
        st.dataframe(b9["sankey"], use_container_width=True)
        st.markdown("**Câu 9.4.4 — Kiểm tra ràng buộc không ngành nào mất quá 5% lao động**")
        st.dataframe(b9["cap_test"], use_container_width=True)
        st.dataframe(b9["cap_table"], use_container_width=True)

# Tab 11 / Bài 10
with tabs[10]:
    st.subheader("🎲 Bài 10 – Two-stage stochastic programming")
    b10 = m.stochastic_programming(budget_b10, reserve_b10, scenario_scale)
    metrics = b10["metrics"].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("SP", f"{metrics['SP']:,.1f}")
    c2.metric("EEV", f"{metrics['EEV']:,.1f}")
    c3.metric("VSS", f"{metrics['VSS']:,.1f}")
    c4.metric("EVPI", f"{metrics['EVPI']:,.1f}")
    c5.metric("Robust worst Z", f"{b10['robust_z']:,.1f}")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(bar_chart(b10["first"], "Hạng mục", "First-stage x", "First-stage here-and-now allocation"), use_container_width=True)
    with col2:
        st.plotly_chart(heatmap_from_matrix(b10["recourse"].set_index("scenario")[["I", "D", "AI", "H"]], "Second-stage recourse theo kịch bản"), use_container_width=True)
    with st.expander("📋 Scenario tree, SP/EEV/WS/VSS/EVPI, wait-and-see và robust allocation"):
        st.markdown("**Scenario tree và β theo từng kịch bản**")
        st.dataframe(b10["scenarios"], use_container_width=True)
        st.markdown("**Bảng chỉ tiêu stochastic programming**")
        st.dataframe(b10["metrics"], use_container_width=True)
        st.markdown("**Second-stage recourse so với expected-value plan**")
        st.dataframe(b10["recourse"], use_container_width=True)
        st.markdown("**Wait-and-see/perfect information allocation**")
        st.dataframe(b10["ws_alloc"], use_container_width=True)
        st.markdown("**Robust allocation theo β xấu nhất**")
        st.dataframe(b10["robust"], use_container_width=True)
        st.markdown("VSS = SP − EEV. EVPI = WS − SP. Nếu VSS dương, mô hình ngẫu nhiên tốt hơn việc chỉ dùng kịch bản kỳ vọng; nếu EVPI dương, thông tin hoàn hảo vẫn có giá trị biên.")

    with st.expander("🧩 Mã Pyomo tham khảo đúng cấu trúc first-stage/second-stage"):
        st.code("\nimport pyomo.environ as pyo\nm = pyo.ConcreteModel()\nm.J = pyo.Set(initialize=['I','D','AI','H'])\nm.S = pyo.Set(initialize=['s1','s2','s3','s4'])\nm.p = pyo.Param(m.S, initialize={'s1':0.30,'s2':0.45,'s3':0.20,'s4':0.05})\nm.beta = pyo.Param(m.S, m.J, initialize=beta_s)\nm.base = pyo.Param(m.J, initialize={'I':1.00,'D':1.10,'AI':1.25,'H':0.95})\nm.x = pyo.Var(m.J, within=pyo.NonNegativeReals)\nm.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)\nm.first_budget = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65000)\nm.recourse_budget = pyo.Constraint(m.S, rule=lambda m,s: sum(m.y[s,j] for j in m.J) <= 15000)\nm.obj = pyo.Objective(expr=sum(m.base[j]*m.x[j] for j in m.J) +\n                      sum(m.p[s]*sum(m.beta[s,j]*m.y[s,j] for j in m.J) for s in m.S),\n                      sense=pyo.maximize)\nsolver = pyo.SolverFactory('glpk')\nsolver.solve(m)\n", language="python")

# Tab 12 / Bài 11
with tabs[11]:
    st.subheader("🤖 Bài 11 – Q-learning cho chính sách kinh tế thích nghi")
    b11 = cached_qlearning(q_episodes, q_lr, q_gamma, int(seed), policy_weights)
    col1, col2 = st.columns([1.2, .8])
    with col1:
        st.plotly_chart(line_chart(b11["curve"], "episode", ["reward", "reward_smoothed"], "Learning curve"), use_container_width=True)
    with col2:
        st.plotly_chart(bar_chart(b11["comparison"], "Policy", "Mean reward", "So sánh chính sách rule-based và Q-learning"), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Số trạng thái", "3⁴ = 81")
    c2.metric("Số hành động", "5")
    c3.metric("Episode horizon", "10 năm")

    with st.expander("🧭 Không gian trạng thái, hành động và kiểm tra chính sách π*(s)"):
        st.markdown("**Không gian trạng thái gồm 4 chỉ số, mỗi chỉ số 3 mức: low / medium / high.**")
        st.dataframe(b11["state_space"], use_container_width=True)
        st.markdown("**5 hành động chính sách đúng theo đề bài**")
        st.dataframe(b11["actions"], use_container_width=True)
        st.markdown("**Kết quả π*(s) tại Việt Nam 2026 thực tế và 4 trạng thái giả định**")
        st.dataframe(b11["state_examples"], use_container_width=True)

    with st.expander("📋 Policy table 81 trạng thái"):
        st.dataframe(b11["policy"], use_container_width=True)
    st.markdown("⚠️ Mô hình Q-learning chỉ minh họa kỹ thuật hỗ trợ ra quyết định; không thay thế trách nhiệm chính trị – xã hội.")

# Tab 13 / Bài 12
with tabs[12]:
    section_header("📡 Bài 12 – So sánh 5 kịch bản S1–S5")
    b12 = cached_scenarios(discount)
    summary12 = b12["summary"].sort_values("GDP_2035", ascending=False)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(bar_chart(summary12, "Kịch bản", "GDP_2035", "GDP 2035 theo 5 kịch bản"), use_container_width=True)
    with c2:
        st.plotly_chart(radar_chart(summary12), use_container_width=True)
    all_paths = []
    for name, path in b12["paths"].items():
        tmp = path[["year", "GDP"]].copy(); tmp["Kịch bản"] = name; all_paths.append(tmp)
    all_paths = pd.concat(all_paths, ignore_index=True)
    st.plotly_chart(px.line(all_paths, x="year", y="GDP", color="Kịch bản", template="plotly_dark", markers=True, title="Đường GDP 2026–2035 theo kịch bản"), use_container_width=True)
    with st.expander("📋 Bảng tổng hợp S1–S5"):
        st.dataframe(summary12, use_container_width=True)

# Tab 14
with tabs[13]:
    section_header("🔎 Phân tích chính sách theo ngành")
    policy = m.sector_policy_analysis(sectors, top_n=9 if exclude_mining else 10)
    col1, col2 = st.columns([1.1, .9])
    with col1:
        st.plotly_chart(bar_chart(policy, "sector_name_vi", "Policy_priority", "Chỉ số ưu tiên chính sách ngành"), use_container_width=True)
    with col2:
        st.dataframe(policy[["sector_name_vi", "Priority", "NetJob_million", "Khuyến nghị"]], use_container_width=True)
    with st.expander("📋 Bảng đầy đủ phân tích ngành"):
        st.dataframe(policy, use_container_width=True)
    st.markdown("✅ Bảng này kết hợp Priority ngành, rủi ro tự động hóa, NetJob và khuyến nghị đầu tư/đào tạo lại.")

# -----------------------------------------------------------------------------
# Export area
# -----------------------------------------------------------------------------

st.divider()
with st.container():
    st.markdown("### 📥 Xuất báo cáo")
    report_md = build_markdown_report(
        "AIDEOM-VN Dashboard Summary",
        {
            "Mô tả": "Dashboard gồm 14 tab: tổng quan dữ liệu, 12 bài mô hình và phân tích chính sách theo ngành.",
            "Nguồn dữ liệu": "vietnam_macro_2020_2025.csv, vietnam_sectors_2024.csv, vietnam_regions_2024.csv.",
            "Gợi ý": "Khi nộp bài, đính kèm link Streamlit Cloud, repo GitHub và ảnh chụp các kết quả chính.",
        },
    )
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Tải báo cáo Markdown", data=report_md, file_name="AIDEOM_VN_report.md", mime="text/markdown", use_container_width=True)
    with col2:
        pdf_bytes = build_pdf_summary({
            "Macro data": macro,
            "Priority ngành": m.priority_index(sectors, weights_b3)["ranking"],
            "S1-S5": cached_scenarios(discount)["summary"],
        })
        st.download_button("⬇️ Tải PDF tóm tắt", data=pdf_bytes, file_name="AIDEOM_VN_summary.pdf", mime="application/pdf", use_container_width=True)
