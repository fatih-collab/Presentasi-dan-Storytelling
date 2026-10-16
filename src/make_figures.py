"""Hasilkan 10 gambar PNG untuk laporan: python src/make_figures.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt

import analysis as A
import charts as C

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

r = A.run_all()
df, s = r["df"], r["sample"]
figs = {
    "01_top_scorers": C.fig_top_scorers(df),
    "02_komposisi_posisi": C.fig_position_composition(s),
    "03_distribusi_gol": C.fig_goal_distribution(s),
    "04_boxplot_posisi": C.fig_box_position(s),
    "05_heatmap_korelasi": C.fig_corr_heatmap(r["corr"]),
    "06_scatter_gol_xg": C.fig_scatter_goal_xg(s),
    "07_gol_per_klub": C.fig_team_goals(df),
    "08_over_under": C.fig_over_under(s),
    "09_diagnostik_residual": C.fig_diagnostics(r["model_final"]),
    "10_forest_koefisien": C.fig_forest(r["model_final_hc3"]),
}
for name, fig in figs.items():
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("ok", name)
