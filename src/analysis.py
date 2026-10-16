"""Modul analisis: pembersihan data, statistik deskriptif, dan uji inferensia.

Hanya bergantung pada pandas, numpy, scipy (tanpa statsmodels) agar mudah
dideploy ke Streamlit Community Cloud.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "epl_2020_21.xlsx"
MIN_MINUTES = 900  # ambang menit bermain (= 10 pertandingan penuh)

POS_LABEL = {"GK": "Kiper", "D": "Bek", "M": "Gelandang", "F": "Penyerang"}


def _position_group(pos: str) -> str:
    """Posisi utama = token pertama selain 'S' (substitute)."""
    tokens = [t for t in str(pos).split() if t != "S"]
    if not tokens:
        return "Cadangan saja"
    return POS_LABEL.get(tokens[0], "Lainnya")


def _fix_mojibake(x: str) -> str:
    """Perbaiki nama yang UTF-8-nya terbaca sebagai latin-1 (mis. 'GÃ¼ndogan')."""
    try:
        return x.encode("latin1").decode("utf8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return x


def load_clean(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.rename(columns={"team_title": "team", "player_name": "player"})
    df["player"] = df["player"].map(_fix_mojibake)
    df["multi_team"] = df["team"].str.contains(",")
    df["pos_group"] = df["position"].map(_position_group)
    df["minutes90"] = df["time"] / 90
    for c in ["goals", "xG", "assists", "xA", "shots", "key_passes"]:
        df[f"{c}_p90"] = df[c] / df["minutes90"]
    df["goal_minus_xG"] = df["goals"] - df["xG"]
    df["conversion"] = np.where(df["shots"] > 0, df["goals"] / df["shots"], np.nan)
    return df


def main_sample(df: pd.DataFrame, min_minutes: int = MIN_MINUTES) -> pd.DataFrame:
    return df[df["time"] >= min_minutes].copy()


# --------------------------------------------------------------------------
# Regresi linear berganda (OLS) manual
# --------------------------------------------------------------------------
def ols(df: pd.DataFrame, y: str, xs: list[str], robust: bool = False) -> dict:
    d = df[[y] + xs].dropna()
    n = len(d)
    X = np.column_stack([np.ones(n), d[xs].to_numpy(float)])
    Y = d[y].to_numpy(float)
    k = X.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    fitted = X @ beta
    resid = Y - fitted
    sse = float(resid @ resid)
    sst = float(((Y - Y.mean()) ** 2).sum())
    ssr = sst - sse
    dfe = n - k
    sigma2 = sse / dfe

    if robust:  # HC3
        h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
        meat = X.T @ (X * (resid / (1 - h))[:, None] ** 2)
        cov = XtX_inv @ meat @ XtX_inv
    else:
        cov = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dfe)
    tcrit = stats.t.ppf(0.975, dfe)

    r2 = ssr / sst
    adj = 1 - (1 - r2) * (n - 1) / dfe
    F = (ssr / (k - 1)) / sigma2
    pF = stats.f.sf(F, k - 1, dfe)

    # beta terstandar
    sd_y = Y.std(ddof=1)
    std_beta = [np.nan] + [beta[i + 1] * d[x].std(ddof=1) / sd_y for i, x in enumerate(xs)]

    coef = pd.DataFrame(
        {
            "koef": beta,
            "std_err": se,
            "t": t,
            "p": p,
            "ci_low": beta - tcrit * se,
            "ci_high": beta + tcrit * se,
            "beta_std": std_beta,
        },
        index=["const"] + xs,
    )

    # VIF
    vif = {}
    for i, x in enumerate(xs):
        others = [c for c in xs if c != x]
        if not others:
            vif[x] = 1.0
            continue
        Xo = np.column_stack([np.ones(n), d[others].to_numpy(float)])
        b = np.linalg.lstsq(Xo, d[x].to_numpy(float), rcond=None)[0]
        r = d[x].to_numpy(float) - Xo @ b
        r2x = 1 - (r @ r) / ((d[x] - d[x].mean()) ** 2).sum()
        vif[x] = 1 / (1 - r2x)

    # Breusch-Pagan (Koenker): regresikan resid^2 pada X
    u2 = resid**2
    b_bp = np.linalg.lstsq(X, u2, rcond=None)[0]
    r_bp = u2 - X @ b_bp
    r2_bp = 1 - (r_bp @ r_bp) / ((u2 - u2.mean()) ** 2).sum()
    lm = n * r2_bp
    p_bp = stats.chi2.sf(lm, k - 1)

    sh_w, sh_p = stats.shapiro(resid)
    dw = float(np.sum(np.diff(resid) ** 2) / sse)

    return {
        "n": n,
        "coef": coef,
        "r2": r2,
        "adj_r2": adj,
        "F": F,
        "pF": pF,
        "df_model": k - 1,
        "df_resid": dfe,
        "rmse": float(np.sqrt(sigma2)),
        "vif": pd.Series(vif),
        "bp_lm": lm,
        "bp_p": p_bp,
        "shapiro_w": sh_w,
        "shapiro_p": sh_p,
        "durbin_watson": dw,
        "fitted": fitted,
        "resid": resid,
        "data": d,
        "robust": robust,
    }


def poisson_glm(df: pd.DataFrame, y: str, xs: list[str], max_iter: int = 50) -> pd.DataFrame:
    """Regresi Poisson (link log) via IRLS, galat baku model-based."""
    d = df[[y] + xs].dropna()
    X = np.column_stack([np.ones(len(d)), d[xs].to_numpy(float)])
    Y = d[y].to_numpy(float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(Y.mean())
    for _ in range(max_iter):
        mu = np.exp(X @ beta)
        W = mu
        z = X @ beta + (Y - mu) / mu
        XtW = X.T * W
        new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    mu = np.exp(X @ beta)
    cov = np.linalg.inv((X.T * mu) @ X)
    se = np.sqrt(np.diag(cov))
    z_ = beta / se
    pv = 2 * stats.norm.sf(np.abs(z_))
    # dispersi Pearson
    disp = float((((Y - mu) ** 2) / mu).sum() / (len(Y) - X.shape[1]))
    out = pd.DataFrame({"koef": beta, "std_err": se, "z": z_, "p": pv,
                        "IRR": np.exp(beta)}, index=["const"] + xs)
    out.attrs["dispersion"] = disp
    return out


def cohens_d_paired(a, b) -> float:
    diff = np.asarray(a) - np.asarray(b)
    return float(diff.mean() / diff.std(ddof=1))


def paired_goal_xg(df: pd.DataFrame) -> dict:
    g, x = df["goals"], df["xG"]
    t = stats.ttest_rel(g, x)
    w = stats.wilcoxon(g, x)
    diff = g - x
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    tc = stats.t.ppf(0.975, len(diff) - 1)
    return {
        "n": len(df),
        "mean_goals": g.mean(),
        "mean_xG": x.mean(),
        "mean_diff": diff.mean(),
        "ci": (diff.mean() - tc * se, diff.mean() + tc * se),
        "t": t.statistic,
        "p_t": t.pvalue,
        "w": w.statistic,
        "p_w": w.pvalue,
        "d": cohens_d_paired(g, x),
    }


def by_position_tests(df: pd.DataFrame, col: str = "goals_p90",
                      groups: tuple[str, ...] = ("Penyerang", "Gelandang", "Bek")) -> dict:
    d = df[df["pos_group"].isin(groups)]
    samples = [d.loc[d["pos_group"] == g, col].to_numpy() for g in groups]
    f = stats.f_oneway(*samples)
    kw = stats.kruskal(*samples)
    # eta kuadrat untuk ANOVA
    grand = d[col].mean()
    ssb = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    sst = ((d[col] - grand) ** 2).sum()
    lev = stats.levene(*samples)
    # Post hoc Mann-Whitney dengan koreksi Bonferroni
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            u = stats.mannwhitneyu(samples[i], samples[j], alternative="two-sided")
            pairs.append((groups[i], groups[j], u.pvalue))
    m = len(pairs)
    pairs = [(a, b, p, min(1.0, p * m)) for a, b, p in pairs]
    return {
        "groups": groups,
        "n": [len(s) for s in samples],
        "means": [s.mean() for s in samples],
        "F": f.statistic,
        "p_F": f.pvalue,
        "eta2": ssb / sst,
        "H": kw.statistic,
        "p_H": kw.pvalue,
        "levene_p": lev.pvalue,
        "posthoc": pairs,
    }


def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["goals", "xG", "assists", "xA", "shots", "key_passes",
            "xGChain", "xGBuildup", "time", "yellow_cards"]
    return df[cols].corr(method="pearson")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df = load_clean()
    s = main_sample(df)
    print("Total:", len(df), "| sampel >=900 menit:", len(s))
    print(s["pos_group"].value_counts().to_string())
    print("\nMulti-team di sampel:", s["multi_team"].sum())
    print("\nDeskriptif sampel:\n", s[["goals", "xG", "assists", "shots", "key_passes", "time"]].describe().round(2).to_string())
    print("\nSkewness goals:", round(s["goals"].skew(), 2), " Shapiro goals:", stats.shapiro(s["goals"]))
    print("\nPaired:", paired_goal_xg(s))
    print("\nPosisi:", by_position_tests(s))
    print("\nKorelasi goals:\n", corr_matrix(s)["goals"].round(3).to_string())
    for xs in (["xG", "shots", "key_passes", "time"], ["xG", "key_passes", "time"]):
        m = ols(s, "goals", xs)
        print("\nOLS", xs, "n=", m["n"])
        print(m["coef"].round(4).to_string())
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
               if k in ("r2", "adj_r2", "F", "pF", "rmse", "bp_lm", "bp_p", "shapiro_w", "shapiro_p", "durbin_watson")})
        print("VIF:", m["vif"].round(2).to_dict())
        mr = ols(s, "goals", xs, robust=True)
        print("HC3 p:", mr["coef"]["p"].round(4).to_dict())


def run_all(df: pd.DataFrame | None = None) -> dict:
    """Jalankan seluruh analisis inti dan kembalikan dalam satu dict."""
    df = load_clean() if df is None else df
    s = main_sample(df)
    full = ["xG", "shots", "key_passes", "time"]
    final = ["xG", "key_passes", "time"]
    return {
        "df": df,
        "sample": s,
        "paired": paired_goal_xg(s),
        "position": by_position_tests(s),
        "corr": corr_matrix(s),
        "model_full": ols(s, "goals", full),
        "model_final": ols(s, "goals", final),
        "model_final_hc3": ols(s, "goals", final, robust=True),
        "poisson": poisson_glm(s, "goals", final),
    }
