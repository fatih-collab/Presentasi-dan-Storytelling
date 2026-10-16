"""Dashboard Streamlit - Analisis Pencetak Gol Premier League 2020/21.

Jalankan lokal : streamlit run app.py
Deploy         : push ke GitHub -> Streamlit Community Cloud -> pilih repo & app.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import analysis as A  # noqa: E402
import charts as C  # noqa: E402
import content as T  # noqa: E402
from charts import fmt  # noqa: E402

st.set_page_config(page_title="Pencetak Gol Premier League 2020/21", page_icon="⚽", layout="wide")


@st.cache_data(show_spinner=False)
def get_results():
    return A.run_all()


def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def team_filter(df: pd.DataFrame, teams: list[str]) -> pd.Series:
    return df["team"].apply(lambda t: any(x in t.split(",") for x in teams))


R = get_results()
DF = R["df"]

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Filter eksplorasi")
st.sidebar.caption("Filter berlaku untuk tab 1–4. Tab Inferensia memakai sampel tetap (≥ 900 menit) agar sesuai laporan.")
min_min = st.sidebar.slider("Minimal menit bermain", 0, 2500, A.MIN_MINUTES, step=100)
pos_opts = [p for p in ["Penyerang", "Gelandang", "Bek", "Kiper", "Cadangan saja"] if p in DF["pos_group"].unique()]
pos_sel = st.sidebar.multiselect("Posisi", pos_opts, default=pos_opts)
team_opts = sorted({t for tt in DF["team"] for t in tt.split(",")})
team_sel = st.sidebar.multiselect("Klub (kosong = semua)", team_opts, default=[])

F = DF[(DF["time"] >= min_min) & DF["pos_group"].isin(pos_sel)]
if team_sel:
    F = F[team_filter(F, team_sel)]

players = sorted(F["player"].unique())
hl = st.sidebar.multiselect("Sorot pemain pada scatter plot", players,
                            default=[p for p in ["Harry Kane", "Mohamed Salah", "Son Heung-Min", "Timo Werner",
                                                 "Jamie Vardy"] if p in players])

# ------------------------------------------------------------------ header
st.title("⚽ Siapa pencetak gol paling efisien? Premier League 2020/21")
st.markdown(
    "Dashboard ini menjelaskan temuan analisis data **522 pemain** Premier League 2020/21 "
    "(sumber: Understat via Kaggle). Alurnya: **orientasi → distribusi → hubungan → perbandingan → inferensia → kesimpulan**. "
    "Ditujukan bagi **dosen dan pembaca akademik** yang ingin memeriksa metode dan angkanya, serta **penggemar sepak bola "
    "dan analis scouting** yang ingin menjelajah data dengan filter di sisi kiri."
)

if len(F) < 5:
    st.warning("Data terlalu sedikit dengan filter ini. Longgarkan filter di sisi kiri.")
    st.stop()

c = st.columns(4)
c[0].metric("Pemain terpilih", f"{len(F)}")
c[1].metric("Total gol", f"{int(F['goals'].sum())}")
c[2].metric("Rata-rata gol per pemain", fmt(F["goals"].mean(), 2))
c[3].metric("Korelasi gol–xG (r)", fmt(F["goals"].corr(F["xG"]), 2))

tabs = st.tabs(["1 · Orientasi", "2 · Distribusi", "3 · Hubungan", "4 · Perbandingan", "5 · Inferensia",
                "6 · Kesimpulan", "Data"])

# ------------------------------------------------------------------ 1 orientasi
with tabs[0]:
    st.subheader("Siapa tokoh utamanya?")
    st.caption("Tujuan: memberi konteks. Siapa pencetak gol terbanyak, dan apakah hasilnya sesuai ekspektasi (xG)?")
    show(C.fig_top_scorers(F))
    show(C.fig_position_composition(F))

# ------------------------------------------------------------------ 2 distribusi
with tabs[1]:
    st.subheader("Seperti apa bentuk datanya?")
    st.caption("Tujuan: memahami distribusi gol dan perbedaan produktivitas antar posisi.")
    show(C.fig_goal_distribution(F))
    groups = tuple(g for g in ("Penyerang", "Gelandang", "Bek") if (F["pos_group"] == g).sum() >= 3)
    if len(groups) >= 2:
        show(C.fig_box_position(F, groups))
    else:
        st.info("Pilih minimal dua posisi (Penyerang/Gelandang/Bek) dengan ≥ 3 pemain untuk melihat perbandingan.")

# ------------------------------------------------------------------ 3 hubungan
with tabs[2]:
    st.subheader("Variabel apa yang berkaitan dengan gol?")
    st.caption("Tujuan: mengidentifikasi hubungan antar variabel dan menjelaskan hubungan utama gol–xG.")
    show(C.fig_corr_heatmap(A.corr_matrix(F)))
    show(C.fig_scatter_goal_xg(F, hl or None))

# ------------------------------------------------------------------ 4 perbandingan
with tabs[3]:
    st.subheader("Siapa dan klub mana yang menonjol?")
    st.caption("Tujuan: membandingkan klub dan menyorot pemain yang menyimpang dari ekspektasi.")
    if F.loc[~F["multi_team"], "team"].nunique() >= 2:
        show(C.fig_team_goals(F))
    else:
        st.info("Pilih minimal dua klub untuk perbandingan antar klub.")
    show(C.fig_over_under(F, n=min(8, max(1, len(F) // 2))))

# ------------------------------------------------------------------ 5 inferensia
with tabs[4]:
    st.subheader("Apakah temuan ini dapat dipercaya secara statistik?")
    st.caption(f"Sampel tetap: {len(R['sample'])} pemain dengan ≥ 900 menit. Taraf signifikansi α = 0,05.")
    n = T.numbers(R)
    pr, po, mf, mh = n["pr"], n["po"], n["mf"], n["mh"]

    st.markdown("#### A. Gol vs xG (uji t berpasangan dan Wilcoxon)")
    st.dataframe(pd.DataFrame({
        "Statistik": ["Rata-rata gol", "Rata-rata xG", "Selisih rata-rata", "CI 95% selisih", "t (uji t berpasangan)",
                      "p (uji t)", "p (Wilcoxon)", "Cohen's d"],
        "Nilai": [fmt(pr["mean_goals"], 2), fmt(pr["mean_xG"], 2), fmt(pr["mean_diff"], 3),
                  f"{fmt(pr['ci'][0], 3)} s.d. {fmt(pr['ci'][1], 3)}", fmt(pr["t"], 2), fmt(pr["p_t"], 3),
                  fmt(pr["p_w"], 3), fmt(pr["d"], 2)]}), hide_index=True, use_container_width=True)
    st.markdown("Uji t tidak signifikan, Wilcoxon signifikan. Distribusi gol miring, sehingga Wilcoxon lebih sesuai; "
                "efeknya kecil.")

    st.markdown("#### B. Gol per 90 menit antar posisi (ANOVA & Kruskal-Wallis)")
    st.dataframe(pd.DataFrame({
        "Posisi": po["groups"], "n": po["n"], "Rata-rata gol/90": [round(x, 3) for x in po["means"]]}),
        hide_index=True, use_container_width=True)
    st.markdown(f"ANOVA F = {fmt(po['F'], 1)}, p {T.pstr(po['p_F'])}, η² = {fmt(po['eta2'], 2)}; "
                f"Kruskal-Wallis H = {fmt(po['H'], 1)}, p {T.pstr(po['p_H'])}. "
                f"Uji Levene p {T.pstr(po['levene_p'])} (varians tidak homogen, sehingga Kruskal-Wallis menjadi penguat).")
    st.dataframe(pd.DataFrame(po["posthoc"], columns=["Posisi A", "Posisi B", "p (Mann-Whitney)", "p (Bonferroni)"])
                 .round(6), hide_index=True, use_container_width=True)

    st.markdown("#### C. Regresi linear berganda: gol ~ xG + umpan kunci + menit bermain")
    coef = mf["coef"].copy()
    coef["p (HC3)"] = mh["coef"]["p"]
    st.dataframe(coef.rename(columns={"koef": "Koefisien", "std_err": "Galat baku", "p": "p (OLS)",
                                      "ci_low": "CI bawah", "ci_high": "CI atas", "beta_std": "β terstandar"})
                 .round(4), use_container_width=True)
    st.markdown(f"R² = {fmt(mf['r2'], 3)}, R² terkoreksi = {fmt(mf['adj_r2'], 3)}, "
                f"F({mf['df_model']}, {mf['df_resid']}) = {fmt(mf['F'], 1)} (p < 0,001), RMSE = {fmt(mf['rmse'], 2)}. "
                f"VIF maksimum = {fmt(mf['vif'].max(), 2)} (tidak ada multikolinearitas serius).")
    show(C.fig_forest(mh))
    show(C.fig_diagnostics(mf))
    with st.expander("Uji ketahanan & alasan `tembakan` dikeluarkan"):
        st.markdown(f"Model awal dengan `tembakan` memiliki VIF tembakan = {fmt(n['vif_shots'], 1)} dan VIF xG = "
                    f"{fmt(n['vif_xg'], 1)}. Model Poisson (uji ketahanan; IRR = exp(koefisien)):")
        st.dataframe(n["pg"].round(4), use_container_width=True)
        st.caption(f"Dispersi Pearson = {fmt(n['pg'].attrs['dispersion'], 2)} (> 1: sedikit overdispersi).")

# ------------------------------------------------------------------ 6 kesimpulan
with tabs[5]:
    st.subheader("Kesimpulan")
    for p in T.kesimpulan(R):
        st.markdown(p)

# ------------------------------------------------------------------ data
with tabs[6]:
    st.subheader("Data terfilter")
    cols = ["player", "team", "pos_group", "games", "time", "goals", "xG", "assists", "xA", "shots", "key_passes",
            "yellow_cards", "red_cards", "npg", "npxG", "xGChain", "xGBuildup", "goals_p90", "goal_minus_xG"]
    st.dataframe(F[cols].sort_values("goals", ascending=False), hide_index=True, use_container_width=True)
    st.download_button("Unduh CSV", F[cols].to_csv(index=False).encode("utf-8"), "epl_2020_21_terfilter.csv",
                       "text/csv")
    st.caption("Sumber data: Understat, melalui dataset Kaggle 'Goal Dataset – Top 5 European Leagues'.")
