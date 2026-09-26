# Laporan TK 1, Soal 2

Isi identitas kelompok, NPM, dan tanda tangan pada `main.tex` sebelum pengumpulan.

## Prasyarat

- Python dengan NumPy, pandas, dan Matplotlib.
- MiKTeX dengan `pdflatex` di `PATH`.

## Reproduksi

Jalankan dari folder ini:

```powershell
python analyze.py
New-Item -ItemType Directory -Force build | Out-Null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

`analyze.py` membaca CSV dari folder `soal2`, menghitung ulang solusi dengan eliminasi Gauss berpivot parsial dan QR Givens yang ditulis sendiri, lalu membuat `metrics.json` serta dua grafik PDF. Hasil kompilasi ada di `build/main.pdf`. Tidak ada solver least squares pustaka yang dipakai untuk menghasilkan koefisien. Dua kali kompilasi menyelesaikan daftar isi PDF dan referensi silang internal.
