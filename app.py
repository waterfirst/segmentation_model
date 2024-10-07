import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import fsolve

st.title("집단 간 네트워크 분절 모델")

st.sidebar.header("파라미터 설정")
α = st.sidebar.slider("α (사망률)", 0.01, 1.0, 0.17, 0.01)
ρ = st.sidebar.slider("ρ (시간 할인율)", 0.01, 1.0, 0.15, 0.01)
δ_prime = st.sidebar.slider("δ' (기본 숙련 프리미엄)", 0.1, 2.0, 0.2, 0.1)
q_prime = st.sidebar.slider("q' (네트워크 외부성 강도)", 1.0, 10.0, 6.9, 0.1)
p = st.sidebar.slider("p (교육기간 네트워크 효과)", 0.1, 5.0, 0.5, 0.1)
c_0 = st.sidebar.slider("c_0 (교육 기본 비용)", 1.0, 10.0, 1.5, 0.1)

st.sidebar.markdown(
    """
### 변수 설명
- α: 사망률 (노동시장 이탈률)
- ρ: 시간 할인율
- δ': 기본 숙련 프리미엄
- q': 네트워크 외부성 강도
- p: 교육기간 네트워크 효과
- c_0: 교육 기본 비용
"""
)


def G(x):
    return norm.cdf(x, loc=1, scale=1.6)


def s_dot_zero(s, Π):
    return 1 - G(c_0 - p * s - Π)


def Π_dot_zero(s):
    return δ_prime + q_prime * s


def find_intersections(s, y1, y2):
    def func(x):
        return np.interp(x, s, y1) - np.interp(x, s, y2)

    roots = fsolve(func, [0.1, 0.5, 0.9])
    return roots[np.abs(func(roots)) < 1e-6]


def plot_graph():
    s = np.linspace(0, 1, 1000)
    Π_zero = Π_dot_zero(s)
    s_zero = np.array([s_dot_zero(si, Πi) for si, Πi in zip(s, Π_zero)])

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(s, Π_zero, "orange", label="Π_t = 0 locus")
    ax.plot(s, s_zero, "blue", label="s_t = 0 locus")

    intersections = find_intersections(s, s_zero, Π_zero)
    labels = ["l", "m", "h"]
    for i, label in zip(range(len(intersections)), labels):
        ax.plot(intersections[i], Π_dot_zero(intersections[i]), "ro")
        ax.annotate(
            f"E_{label}",
            (intersections[i], Π_dot_zero(intersections[i])),
            xytext=(5, 5),
            textcoords="offset points",
        )

    def trajectory(s0, Π0, direction, steps=1000):
        path_s, path_Π = [s0], [Π0]
        for _ in range(steps):
            ds = α * (s_dot_zero(path_s[-1], path_Π[-1]) - path_s[-1])
            dΠ = (α + ρ) * (Π_dot_zero(path_s[-1]) - path_Π[-1])
            path_s.append(path_s[-1] + direction * ds * 0.01)
            path_Π.append(path_Π[-1] + direction * dΠ * 0.01)
            if not (0 <= path_s[-1] <= 1):
                break
        return path_s, path_Π

    if len(intersections) > 1:
        opt_s, opt_Π = trajectory(intersections[1], Π_dot_zero(intersections[1]), 1)
        ax.plot(opt_s, opt_Π, "g-", label="Optimistic Path")

        pes_s, pes_Π = trajectory(intersections[1], Π_dot_zero(intersections[1]), -1)
        ax.plot(pes_s, pes_Π, "r-", label="Pessimistic Path")

    ax.set_xlabel("s_t (Skilled Person Rate)")
    ax.set_ylabel("Π_t (Expected return on human capital investment)")
    ax.set_title("Intergroup network segmentation model")
    ax.legend()
    ax.grid(True)

    if len(intersections) > 2:
        ax.axvline(x=intersections[0], color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=intersections[2], color="gray", linestyle="--", alpha=0.5)
        ax.annotate(
            "Range of Indeterminacy",
            xy=(0.4, 0.1),
            xytext=(0.4, 0.05),
            arrowprops=dict(arrowstyle="<->"),
            ha="center",
        )

    # 점선 추가
    for i in np.arange(0, 10, 0.5):
        ax.axhline(y=i, color="gray", linestyle=":", alpha=0.3)
    for i in np.arange(0, 1, 0.1):
        ax.axvline(x=i, color="gray", linestyle=":", alpha=0.3)

    ax.set_ylim(0, 10)
    ax.set_xlim(0, 1)

    return fig


st.pyplot(plot_graph())

st.markdown(
    """
## 시뮬레이션 설명

이 모델은 집단 간 네트워크 분절이 인적자본 투자와 숙련도에 미치는 영향을 시뮬레이션합니다.

- **파란색 선 (s_t = 0 locus)**: 숙련자 비율의 변화가 없는 상태를 나타냅니다.
- **주황색 선 (Π_t = 0 locus)**: 인적자본 투자의 기대 수익 변화가 없는 상태를 나타냅니다.
- **녹색 선 (Optimistic Path)**: 긍정적 기대에 따른 발전 경로를 보여줍니다.
- **빨간색 선 (Pessimistic Path)**: 부정적 기대에 따른 쇠퇴 경로를 보여줍니다.
- **Range of Indeterminacy**: 초기 조건에 따라 결과가 달라질 수 있는 불확정 구간을 나타냅니다.

파라미터를 조정하여 다양한 시나리오를 탐색해보세요. 네트워크 외부성의 강도(q')나 교육 비용(c_0) 등의 변화가 모델의 동태에 어떤 영향을 미치는지 관찰할 수 있습니다.
"""
)
