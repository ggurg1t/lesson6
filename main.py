import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 연간 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")
st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온의 추세선(단순 선형회귀)으로 원하는 해의 기온을 예상합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly(url: str):
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 기준 기간 이후 데이터 제외

    grouped = df.groupby("연도")["평균기온"].agg(연평균기온="mean", 관측일수="count")
    grouped = grouped[grouped["관측일수"] >= MIN_DAYS]  # 관측일 300일 미만 해 제외
    return grouped.reset_index()


try:
    yearly = load_yearly(DATA_URL)
except Exception as e:  # 네트워크/파싱 오류 안내
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 만들 수 있는 연도가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# ---- 입력 ----
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1rem 0;">
      <div style="font-size:1.2rem; color:gray;">{year}년의 예상 연평균기온</div>
      <div style="font-size:4.5rem; font-weight:700;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- 데이터 정보 ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("회귀에 쓴 연도 수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")
st.write(
    f"회귀식: 연평균기온 = {slope:.4f} × 연도 + ({intercept:.2f})  "
    f"(연간 약 {slope:+.4f}℃ 변화)"
)
st.caption(
    f"{LAST_YEAR}년까지의 데이터만 사용했고, 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다."
)

# ---- 그래프 ----
x_line = np.array([min(1900, start_year), 2100], dtype=float)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8),
        customdata=yearly["관측일수"],
        hovertemplate="%{x}년<br>%{y:.2f}℃<br>관측일 %{customdata}일<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_line,
        y=slope * x_line + intercept,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="black")),
        hovertemplate="%{x}년 예상<br>%{y:.2f}℃<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온 산점도와 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=520,
    legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(yearly.round(2), use_container_width=True, hide_index=True)
