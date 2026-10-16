# Analisis Pencetak Gol Premier League 2020/21

Dashboard Streamlit dan kode analisis untuk tugas analisis data (eksplorasi, inferensia, visualisasi).
Sumber data: Understat melalui dataset Kaggle "Goal Dataset – Top 5 European Leagues" (EPL 2020/21, 522 pemain).

## Struktur
```
app.py              # dashboard Streamlit
src/analysis.py     # pembersihan data, uji t/Wilcoxon, ANOVA/Kruskal-Wallis, regresi OLS (manual) & Poisson
src/charts.py       # 10 grafik (matplotlib)
src/content.py      # teks interpretasi & kesimpulan (angka diambil dari hasil analisis)
src/make_figures.py # menghasilkan PNG di folder figures/ untuk laporan
data/epl_2020_21.xlsx
```

## Menjalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Buat repository GitHub, lalu push seluruh isi folder ini (pastikan `data/epl_2020_21.xlsx` ikut).
2. Buka https://share.streamlit.io, login dengan GitHub, klik **Create app**.
3. Pilih repository, branch `main`, dan *Main file path* = `app.py`, lalu **Deploy**.
