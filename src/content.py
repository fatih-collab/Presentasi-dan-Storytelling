"""Teks naratif (interpretasi & kesimpulan) - satu sumber untuk dashboard dan laporan.

Semua angka diambil dari hasil analisis (run_all), tidak diketik manual.
"""
from __future__ import annotations

from charts import fmt


def pstr(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"= {fmt(p, 3)}"


def numbers(r: dict) -> dict:
    s, pr, po = r["sample"], r["paired"], r["position"]
    mf, mh, mfull, pg = r["model_final"], r["model_final_hc3"], r["model_full"], r["poisson"]
    top = r["df"].nlargest(2, "goals")
    d = s.assign(diff=s["goals"] - s["xG"])
    hi, lo = d.loc[d["diff"].idxmax()], d.loc[d["diff"].idxmin()]
    return dict(
        n_all=len(r["df"]), n=len(s), median=s["goals"].median(), mean=s["goals"].mean(),
        skew=s["goals"].skew(), k1=top.iloc[0], k2=top.iloc[1], pr=pr, po=po, mf=mf, mh=mh,
        mfull=mfull, pg=pg, hi=hi, lo=lo,
        vif_shots=mfull["vif"]["shots"], vif_xg=mfull["vif"]["xG"],
        b_xg=mf["coef"].loc["xG", "koef"], ci_xg=(mh["coef"].loc["xG", "ci_low"], mh["coef"].loc["xG", "ci_high"]),
    )


def kesimpulan(r: dict) -> list[str]:
    n = numbers(r)
    pr, po, mf, mh, pg = n["pr"], n["po"], n["mf"], n["mh"], n["pg"]
    m = dict(zip(po["groups"], po["means"]))
    kp = mh["coef"].loc["key_passes"]
    return [
        # 1. gambaran umum
        f"Analisis terhadap {n['n']} pemain Premier League musim 2020/21 yang bermain minimal 900 menit "
        f"(dari {n['n_all']} pemain dalam data) menunjukkan bahwa produksi gol sangat terkonsentrasi pada sedikit "
        f"pemain. Median gol per pemain hanya {fmt(n['median'], 0)} dan rata-ratanya {fmt(n['mean'], 2)}, dengan "
        f"skewness {fmt(n['skew'], 2)} yang menandakan distribusi miring ke kanan. Pencetak gol terbanyak adalah "
        f"{n['k1']['player']} ({int(n['k1']['goals'])} gol) dan {n['k2']['player']} ({int(n['k2']['goals'])} gol).",
        # 2. uji berpasangan
        f"Pada uji gol versus xG, rata-rata gol ({fmt(pr['mean_goals'], 2)}) sedikit lebih rendah daripada rata-rata "
        f"xG ({fmt(pr['mean_xG'], 2)}), dengan selisih {fmt(pr['mean_diff'], 2)} gol (interval kepercayaan 95%: "
        f"{fmt(pr['ci'][0], 2)} hingga {fmt(pr['ci'][1], 2)}). Uji t berpasangan tidak signifikan "
        f"(p {pstr(pr['p_t'])}), sedangkan uji Wilcoxon signifikan (p {pstr(pr['p_w'])}). Karena distribusi gol tidak "
        f"normal, uji Wilcoxon lebih sesuai, tetapi ukuran efeknya kecil (Cohen's d = {fmt(pr['d'], 2)}). Dengan demikian "
        f"pemain secara umum mencetak gol sedikit di bawah ekspektasi, namun selisih praktisnya kecil sehingga xG "
        f"merupakan acuan yang cukup baik untuk kelompok pemain. Pada tingkat individu penyimpangannya besar, misalnya "
        f"{n['hi']['player']} ({'+' if n['hi']['diff'] > 0 else ''}{fmt(n['hi']['diff'], 1)} gol di atas xG) dan "
        f"{n['lo']['player']} ({fmt(n['lo']['diff'], 1)} gol), yang dapat mencerminkan kualitas penyelesaian akhir, "
        f"keberuntungan, atau variasi acak dalam satu musim.",
        # 3. posisi
        f"Produktivitas gol per 90 menit berbeda sangat nyata antar posisi (ANOVA: F = {fmt(po['F'], 1)}, "
        f"p {pstr(po['p_F'])}; Kruskal-Wallis: H = {fmt(po['H'], 1)}, p {pstr(po['p_H'])}), dengan η² = "
        f"{fmt(po['eta2'], 2)}, artinya posisi saja menjelaskan sekitar {fmt(100 * po['eta2'], 0)}% variasi. Rata-rata "
        f"penyerang adalah {fmt(m['Penyerang'], 2)} gol per 90 menit, gelandang {fmt(m['Gelandang'], 2)}, dan bek "
        f"{fmt(m['Bek'], 2)}. Seluruh pasangan posisi berbeda signifikan setelah koreksi Bonferroni.",
        # 4. regresi
        f"Model regresi linear berganda akhir (gol dengan prediktor xG, umpan kunci, dan menit bermain) menjelaskan "
        f"{fmt(100 * mf['r2'], 1)}% variasi gol (R² = {fmt(mf['r2'], 3)}; F({mf['df_model']}, {mf['df_resid']}) = "
        f"{fmt(mf['F'], 1)}, p < 0,001). Setelah memakai galat baku robust (HC3), hanya xG yang berpengaruh signifikan: "
        f"setiap tambahan 1 xG berasosiasi dengan sekitar {fmt(n['b_xg'], 2)} gol tambahan (interval kepercayaan 95%: "
        f"{fmt(n['ci_xg'][0], 2)} hingga {fmt(n['ci_xg'][1], 2)}). Umpan kunci (p {pstr(kp['p'])}) dan menit bermain "
        f"tidak signifikan. Variabel tembakan sengaja dikeluarkan karena multikolinearitas yang tinggi dengan xG "
        f"(VIF tembakan = {fmt(n['vif_shots'], 1)}, VIF xG = {fmt(n['vif_xg'], 1)}), sebab xG sendiri dihitung dari "
        f"kualitas tembakan.",
        # 5. keterbatasan
        f"Hasil ini perlu dibaca dengan hati-hati. Asumsi klasik regresi tidak sepenuhnya terpenuhi: residual tidak "
        f"berdistribusi normal (Shapiro-Wilk p {pstr(mf['shapiro_p'])}) dan variansnya tidak konstan (Breusch-Pagan "
        f"p {pstr(mf['bp_p'])}), sehingga inferensia dilakukan dengan galat baku robust. Model Poisson sebagai uji "
        f"ketahanan tetap menempatkan xG sebagai prediktor yang sangat signifikan (rasio laju = "
        f"{fmt(pg.loc['xG', 'IRR'], 2)}), tetapi menilai umpan kunci dan menit bermain signifikan, sehingga "
        f"kesimpulan tentang prediktor selain xG bergantung pada spesifikasi model. Selain itu data hanya mencakup "
        f"satu musim, hubungan yang ditemukan bersifat asosiasi dan bukan sebab-akibat, xG merupakan estimasi model, "
        f"dan gol penalti belum dipisahkan.",
        # 6. implikasi
        f"Secara praktis, bagi analis dan pemandu bakat, gol per 90 menit dan xG lebih informatif daripada total gol "
        f"karena menyesuaikan waktu bermain dan kualitas peluang; penyimpangan besar dari xG umumnya sulit "
        f"dipertahankan antarmusim (regresi ke rata-rata), meskipun hal ini tidak diuji pada data satu musim ini. Penelitian lanjutan disarankan memakai data beberapa "
        f"musim, model data cacah (Poisson atau binomial negatif) dengan efek acak per klub, serta memisahkan gol "
        f"penalti (npxG) dan variabel konteks seperti kekuatan lawan.",
    ]
