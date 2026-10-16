"""Fungsi pembuat grafik (matplotlib) - dipakai bersama oleh dashboard & laporan.

Prinsip (visualisasi eksplanatori): judul = pesan utama, warna hanya untuk
menyorot (abu-abu = konteks), label langsung pada data, grid minimal.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis import ols

# --- Palet (referensi tervalidasi: slot kategorikal 1-3, abu-abu netral, diverging biru<->merah)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED = "#e34948"
GRAY, GRAY_D = "#c9c8c2", "#8a8983"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
POS_COLOR = {"Penyerang": BLUE, "Gelandang": ORANGE, "Bek": AQUA, "Kiper": GRAY_D}

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def fmt(x: float, d: int = 1) -> str:
    """Format angka gaya Indonesia (koma desimal)."""
    return f"{x:.{d}f}".replace(".", ",")


def _title(ax, title: str, subtitle: str | None = None):
    extra = 14 * title.count("\n")
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK,
                 pad=(30 if subtitle else 12) + extra)
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def _comma_axis(ax, which="y"):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(f)


def _grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _source(fig, text="Sumber: Understat via Kaggle, Premier League 2020/21. Pengolahan penulis."):
    fig.text(0.01, -0.01, text, fontsize=7.5, color=GRAY_D, ha="left", va="top")


KEEP_FULL = {"Son Heung-Min"}


def _short(name: str) -> str:
    parts = name.split()
    if name in KEEP_FULL:
        return name
    return name if len(parts) == 1 else f"{parts[0][0]}. {' '.join(parts[1:])}"


# ---------------------------------------------------------------- 1
def fig_top_scorers(df: pd.DataFrame, n: int = 10):
    t = df.nlargest(n, "goals").iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(t))
    ax.barh(y, t["goals"], height=0.55, color=BLUE, zorder=2)
    ax.scatter(t["xG"], y, marker="D", s=46, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=3)
    for yi, (g, x) in enumerate(zip(t["goals"], t["xG"])):
        ax.text(max(g, x) + 0.5, yi, f"{int(g)}  (xG {fmt(x)})", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [f"{p}  ·  {tm}" for p, tm in zip(t["player"], t["team"].str.replace("Manchester ", "Man. "))],
                  fontsize=9)
    ax.set_xlim(0, t["goals"].max() * 1.28)
    _grid(ax, "x")
    ax.set_xlabel("Gol")
    top = df.nlargest(2, "goals")
    _title(ax, f"{top.iloc[0]['player']} ({int(top.iloc[0]['goals'])}) dan {top.iloc[1]['player']} "
               f"({int(top.iloc[1]['goals'])}) memimpin daftar pencetak gol",
           "Batang biru = gol; berlian oranye = gol yang diharapkan (xG)")
    _source(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 2
def fig_position_composition(sample: pd.DataFrame):
    order = ["Bek", "Penyerang", "Gelandang", "Kiper"]
    c = sample["pos_group"].value_counts().reindex(order).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    cols = [BLUE if k == c.idxmax() else GRAY for k in c.index]
    bars = ax.bar(c.index, c.values, color=cols, width=0.55, zorder=2)
    for b, v in zip(bars, c.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{int(v)}  ({fmt(100 * v / c.sum(), 0)}%)",
                ha="center", fontsize=9, color=INK2, bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=4)
    ax.set_ylim(0, c.max() * 1.18)
    ax.set_ylabel("Jumlah pemain")
    _grid(ax, "y")
    _title(ax, f"{c.idxmax()} mendominasi sampel: {int(c.max())} dari {int(c.sum())} pemain",
           "Pemain dengan ≥ 900 menit bermain, menurut posisi utama")
    _source(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 3
def fig_goal_distribution(sample: pd.DataFrame):
    g = sample["goals"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.arange(-0.5, g.max() + 1.5, 1)
    ax.hist(g, bins=bins, color=BLUE, edgecolor="white", linewidth=1.2, zorder=2)
    ax.axvline(g.median(), color=INK, linestyle=(0, (4, 3)), linewidth=1.2, zorder=3)
    ax.axvline(g.mean(), color=ORANGE, linewidth=1.6, zorder=3)
    ymax = ax.get_ylim()[1]
    ax.text(g.median() + 0.25, ymax * 0.90, f"median\n= {fmt(g.median(), 0)}", ha="left", va="top", fontsize=9, color=INK)
    ax.text(g.mean() + 0.35, ymax * 0.62, f"rata-rata = {fmt(g.mean(), 2)}", ha="left", fontsize=9, color=ORANGE,
            fontweight="bold")
    ax.set_xlabel("Gol dalam satu musim")
    ax.set_ylabel("Jumlah pemain")
    _grid(ax, "y")
    _title(ax, f"Separuh pemain mencetak ≤ {int(g.median())} gol; distribusi sangat miring ke kanan",
           f"Skewness = {fmt(g.skew(), 2)} (n = {len(g)}); sedikit pemain mencetak banyak gol")
    _source(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 4
def fig_box_position(sample: pd.DataFrame, groups=("Penyerang", "Gelandang", "Bek")):
    d = sample[sample["pos_group"].isin(groups)]
    data = [d.loc[d["pos_group"] == g, "goals_p90"].to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    rng = np.random.default_rng(7)
    for i, (g, v) in enumerate(zip(groups, data)):
        col = POS_COLOR[g]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=col, alpha=0.35, linewidths=0, zorder=2)
        bp = ax.boxplot(v, positions=[i], widths=0.38, showfliers=False, patch_artist=True, zorder=3,
                        medianprops=dict(color=INK, linewidth=1.8),
                        boxprops=dict(facecolor="none", edgecolor=col, linewidth=1.6),
                        whiskerprops=dict(color=col, linewidth=1.4), capprops=dict(color=col, linewidth=1.4))
    ax.set_xticks(range(len(groups)), [f"{g}\nn = {len(v)} · rata-rata {fmt(v.mean(), 2)}" for g, v in zip(groups, data)])
    ax.set_ylabel("Gol per 90 menit")
    ax.set_ylim(0, max(v.max() for v in data) * 1.08)
    _comma_axis(ax, "y")
    _grid(ax, "y")
    if "Penyerang" in groups and "Bek" in groups:
        a, b = data[groups.index("Penyerang")], data[groups.index("Bek")]
        ttl = f"Penyerang rata-rata {fmt(a.mean() / b.mean())} kali lebih produktif daripada bek"
    else:
        ttl = "Produktivitas gol per 90 menit berbeda menurut posisi"
    _title(ax, ttl,
           "Gol per 90 menit menurut posisi; titik = pemain, kotak = kuartil, garis hitam = median")
    _source(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 5
def fig_corr_heatmap(corr: pd.DataFrame):
    labels = {"goals": "Gol", "xG": "xG", "assists": "Assist", "xA": "xA", "shots": "Tembakan",
              "key_passes": "Umpan kunci", "xGChain": "xGChain", "xGBuildup": "xGBuildup",
              "time": "Menit", "yellow_cards": "Kartu kuning"}
    c = corr.rename(index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(c.values, cmap=_div_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(c)), c.index)
    for i in range(len(c)):
        for j in range(len(c)):
            v = c.values[i, j]
            ax.text(j, i, fmt(v, 2), ha="center", va="center", fontsize=7.5,
                    color="white" if abs(v) > 0.62 else INK)
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-.5, len(c)), minor=True)
    ax.set_yticks(np.arange(-.5, len(c)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("Korelasi Pearson (r)")
    _title(ax, f"Gol paling erat berkaitan dengan xG (r = {fmt(corr.loc['goals', 'xG'], 2)}) "
               f"dan tembakan (r = {fmt(corr.loc['goals', 'shots'], 2)})",
           "Korelasi Pearson antar variabel kinerja, pemain ≥ 900 menit")
    _source(fig)
    fig.tight_layout()
    return fig


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div", [RED, "#f0efec", BLUE])


# ---------------------------------------------------------------- 6
def fig_scatter_goal_xg(sample: pd.DataFrame, highlight: list[str] | None = None):
    m = ols(sample, "goals", ["xG"])
    b0, b1 = m["coef"]["koef"]
    x = sample["xG"].to_numpy()
    xs = np.linspace(0, x.max(), 100)
    n = len(sample)
    s2 = m["rmse"] ** 2
    se_fit = np.sqrt(s2 * (1 / n + (xs - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()))
    tc = stats.t.ppf(0.975, n - 2)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.fill_between(xs, b0 + b1 * xs - tc * se_fit, b0 + b1 * xs + tc * se_fit, color=BLUE, alpha=0.15, linewidth=0)
    ax.plot([0, x.max()], [0, x.max()], color=GRAY_D, linestyle=(0, (4, 3)), linewidth=1)
    ax.scatter(sample["xG"], sample["goals"], s=22, color=GRAY_D, alpha=0.55, linewidths=0, zorder=2)
    ax.plot(xs, b0 + b1 * xs, color=BLUE, linewidth=2.2, zorder=3)
    ax.text(x.max() * 0.74, (b0 + b1 * x.max() * 0.74) - 3.1, f"garis regresi: gol = {fmt(b0, 2)} + {fmt(b1, 2)}·xG",
            fontsize=8.5, color=BLUE, ha="left", fontweight="bold")
    lab = highlight or ["Harry Kane", "Mohamed Salah", "Son Heung-Min", "Timo Werner", "Jamie Vardy",
                        "Gareth Bale", "Neal Maupay"]
    offs = {"Harry Kane": (-8, 6, "right"), "Mohamed Salah": (-9, -3, "right"), "Son Heung-Min": (-8, 5, "right"),
            "Timo Werner": (8, -4, "left"), "Jamie Vardy": (8, -6, "left"), "Gareth Bale": (-8, 5, "right"),
            "Neal Maupay": (8, 3, "left")}
    for p in lab:
        r = sample[sample["player"] == p]
        if r.empty:
            continue
        r = r.iloc[0]
        dx, dy, ha = offs.get(p, (6, 4, "left"))
        ax.scatter([r["xG"]], [r["goals"]], s=34, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=4)
        ax.annotate(_short(p), (r["xG"], r["goals"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=INK, ha=ha, zorder=5)
    ax.set_xlabel("Gol yang diharapkan (xG)")
    ax.set_ylabel("Gol")
    ax.set_xlim(-0.5, x.max() * 1.05)
    ax.set_ylim(-1, sample["goals"].max() * 1.08)
    _grid(ax, "both")
    _title(ax, f"xG menjelaskan sebagian besar variasi gol (R² = {fmt(m['r2'], 2)})",
           "Tiap titik = satu pemain; area biru = interval kepercayaan 95% garis regresi; garis putus = gol sama dengan xG")
    _source(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 7
def fig_team_goals(df: pd.DataFrame):
    d = df[~df["multi_team"]]
    g = d.groupby("team")["goals"].sum().sort_values()
    top = d.loc[d.groupby("team")["goals"].idxmax()].set_index("team")
    share = (top["goals"] / g).reindex(g.index)
    fig, ax = plt.subplots(figsize=(8, 6.2))
    cols = [BLUE if t == g.idxmax() else GRAY for t in g.index]
    ax.barh(g.index, g.values, color=cols, height=0.62, zorder=2)
    for i, (t, v) in enumerate(g.items()):
        ax.text(v + 0.8, i, f"{int(v)}   {_short(top.loc[t, 'player'])} ({fmt(100 * share[t], 0)}%)",
                va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, g.max() * 1.42)
    ax.set_xlabel("Total gol pemain klub")
    ax.tick_params(axis="y", labelsize=8.5)
    _grid(ax, "x")
    lo, hi = share.idxmin(), share.idxmax()
    _title(ax, f"{g.idxmax()} mencetak gol terbanyak ({int(g.max())}); {hi} paling bergantung\n"
               f"pada satu pencetak gol ({fmt(100 * share[hi], 0)}%), {lo} paling merata ({fmt(100 * share[lo], 0)}%)",
           "Total gol pemain per klub; di kanan batang: pencetak gol terbanyak dan porsinya terhadap gol klub")
    _source(fig, "Sumber: Understat via Kaggle. Pemain yang berpindah klub tidak dihitung; total dapat berbeda dari gol resmi klub.")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 8
def fig_over_under(sample: pd.DataFrame, n: int = 8):
    d = sample.assign(diff=sample["goals"] - sample["xG"])
    sel = pd.concat([d.nsmallest(n, "diff"), d.nlargest(n, "diff")]).sort_values("diff")
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    cols = [RED if v < 0 else BLUE for v in sel["diff"]]
    ax.barh(range(len(sel)), sel["diff"], color=cols, height=0.62, zorder=2)
    ax.axvline(0, color=INK2, linewidth=1)
    for i, (_, r) in enumerate(sel.iterrows()):
        v = r["diff"]
        txt = f"{'+' if v > 0 else '−'}{fmt(abs(v))}   ({int(r['goals'])} gol vs xG {fmt(r['xG'])})"
        ax.text(v + (0.15 if v > 0 else -0.15), i, txt, va="center", ha="left" if v > 0 else "right",
                fontsize=8, color=INK2)
    ax.set_yticks(range(len(sel)), [f"{p}" for p in sel["player"]], fontsize=8.5)
    lim = max(abs(sel["diff"]).max() * 1.9, 1)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Gol − xG (positif = di atas ekspektasi)")
    _grid(ax, "x")
    hi, lo = d.loc[d["diff"].idxmax()], d.loc[d["diff"].idxmin()]
    _title(ax, f"{hi['player']} paling melampaui xG (+{fmt(hi['diff'])}), {lo['player']} paling jauh di bawahnya (−{fmt(abs(lo['diff']))})",
           f"{n} pemain di atas dan {n} di bawah ekspektasi (biru = di atas xG, merah = di bawah xG)")
    _source(fig)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- 9
def fig_diagnostics(model: dict):
    fit, res = model["fitted"], model["resid"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    ax.scatter(fit, res, s=16, color=GRAY_D, alpha=0.6, linewidths=0, zorder=2)
    ax.axhline(0, color=BLUE, linewidth=1.6)
    ax.set_xlabel("Nilai prediksi (gol)")
    ax.set_ylabel("Residual")
    _grid(ax, "y")
    ax.set_title("Residual vs prediksi", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0.02, 0.95, f"Breusch-Pagan p {_p(model['bp_p'])}", transform=ax.transAxes, fontsize=8.5,
            color=INK2, va="top")
    ax = axes[1]
    (osm, osr), (sl, ic, _) = stats.probplot(res, dist="norm")
    ax.scatter(osm, osr, s=16, color=GRAY_D, alpha=0.6, linewidths=0, zorder=2)
    ax.plot(osm, sl * np.asarray(osm) + ic, color=BLUE, linewidth=1.6)
    ax.set_xlabel("Kuantil teoretis normal")
    ax.set_ylabel("Kuantil residual")
    _grid(ax, "y")
    ax.set_title("Q-Q plot residual", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0.02, 0.95, f"Shapiro-Wilk p {_p(model['shapiro_p'])}", transform=ax.transAxes, fontsize=8.5,
            color=INK2, va="top")
    fig.suptitle("Sebaran residual tidak konstan dan berekor tebal: asumsi OLS tidak terpenuhi",
                 x=0.01, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _source(fig)
    return fig


def _p(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"= {fmt(p, 3)}"


# ---------------------------------------------------------------- 10
def fig_forest(model_hc3: dict, labels: dict | None = None):
    labels = labels or {"xG": "xG", "key_passes": "Umpan kunci", "time": "Menit bermain", "shots": "Tembakan"}
    d, c = model_hc3["data"], model_hc3["coef"].drop("const")
    sd_y = d["goals"].std(ddof=1)
    rows = []
    for k in c.index:
        sc = d[k].std(ddof=1) / sd_y
        rows.append((labels.get(k, k), c.loc[k, "koef"] * sc, c.loc[k, "ci_low"] * sc, c.loc[k, "ci_high"] * sc,
                     c.loc[k, "p"]))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (lab, b, lo, hi, p) in enumerate(rows):
        sig = p < 0.05
        col = BLUE if sig else GRAY_D
        ax.plot([lo, hi], [i, i], color=col, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([b], [i], s=62, color=col, edgecolor="white", linewidth=1.4, zorder=3)
        ax.text(hi + 0.03, i + 0.22, f"β = {fmt(b, 2)}   p {_p(p)}", fontsize=8.5, color=INK2, va="bottom")
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Koefisien terstandar (β) dengan interval kepercayaan 95%")
    ax.set_xlim(min(r[2] for r in rows) - 0.08, max(r[3] for r in rows) + 0.5)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    _comma_axis(ax, "x")
    _grid(ax, "x")
    _title(ax, "Setelah galat baku robust, hanya xG yang berpengaruh signifikan terhadap gol",
           "Biru = signifikan (p < 0,05); abu-abu = tidak signifikan. Galat baku HC3, n = " + str(model_hc3["n"]))
    _source(fig)
    fig.tight_layout()
    return fig


ALL_FIGS = [
    ("01_top_scorers", "Pencetak gol terbanyak"),
    ("02_komposisi_posisi", "Komposisi sampel menurut posisi"),
    ("03_distribusi_gol", "Distribusi jumlah gol per pemain"),
    ("04_boxplot_posisi", "Gol per 90 menit menurut posisi"),
    ("05_heatmap_korelasi", "Matriks korelasi antar variabel"),
    ("06_scatter_gol_xg", "Hubungan gol dan xG"),
    ("07_gol_per_klub", "Total gol pemain per klub"),
    ("08_over_under", "Selisih gol dan xG (over/underperformer)"),
    ("09_diagnostik_residual", "Diagnostik residual model regresi"),
    ("10_forest_koefisien", "Koefisien regresi terstandar"),
]
