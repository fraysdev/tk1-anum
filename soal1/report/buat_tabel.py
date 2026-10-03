"""Membuat potongan LaTeX (baris tabel dan listing lampiran) dari ../hasil/*.csv dan ../soal1.ipynb.

Jalankan ulang setelah notebook dijalankan ulang, supaya angka di soal1.tex selalu sama dengan isi hasil/.
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
HASIL = ROOT.parent / 'hasil'
NOTEBOOK = ROOT.parent / 'soal1.ipynb'
OUT = ROOT / 'generated'
OUT.mkdir(exist_ok=True)


def sci(x: float, digit: int = 2) -> str:
    # 1.11e-16 -> $1.11\times10^{-16}$
    mantisa, pangkat = f'{x:.{digit}e}'.split('e')
    return rf'${mantisa}\times10^{{{int(pangkat)}}}$'


def ribuan(n: int) -> str:
    return f'{n:,}'.replace(',', r'\,')


def tulis(nama: str, baris: list[str]) -> None:
    (OUT / nama).write_text('\n'.join(baris) + '\n', encoding='utf-8')


# Tabel 2.1: validasi
v = pd.read_csv(HASIL / 'validasi.csv')
baris = []
for r in v.itertuples():
    minimum = sci(r.min_taknol_luar_diag) if r.min_taknol_luar_diag < 1e-3 else f'{r.min_taknol_luar_diag:.4f}'
    baris.append(f'{r.N} & {sci(r.galat_baris_maks)} & {r.min_diag:.3f} & {minimum} & '
                 f'{"Ya" if r.valid else "Tidak"} \\\\')
tulis('tabel_validasi.tex', baris)

# Tabel 4.1: penyimpanan
s = pd.read_csv(HASIL / 'storage.csv')
tulis('tabel_storage.tex', [
    f'{r.N} & {ribuan(r.dense_byte)} & {ribuan(r.band_byte)} & {ribuan(r.band_pivot_byte)} & '
    f'{r.rasio_dense_per_band_pivot:.1f} \\\\' for r in s.itertuples()])

# Tabel 6.1: benchmark (data soal saja)
b = pd.read_csv(HASIL / 'benchmark_ringkas.csv')
b = b[b['sumber'] == 'data']
tulis('tabel_benchmark.tex', [
    f'{r.N} & {r.baca_csv_ms:.2f} & {r.prep_dense_ms:.2f} & {r.solver_dense_ms:.2f} & {r.prep_band_ms:.2f} & '
    f'{r.solver_band_ms:.2f} & {r.nbytes_dense_KiB:.1f} & {r.nbytes_band_KiB:.1f} & {r.peak_dense_KiB:.1f} & '
    f'{r.peak_band_KiB:.1f} \\\\' for r in b.itertuples()])

# Tabel 6.2: kemiringan log-log
k = pd.read_csv(HASIL / 'kemiringan_loglog.csv', index_col=0)
nama = {
    'waktu dense (teori 3)': ('Waktu solver dense', 3), 'waktu band (teori 1)': ('Waktu solver banded', 1),
    'flop faktorisasi dense (teori 3)': ('Flop faktorisasi dense', 3),
    'flop faktorisasi banded (teori 1)': ('Flop faktorisasi banded', 1),
    'nbytes dense (teori 2)': (r'\texttt{.nbytes} dense', 2), 'nbytes band (teori 1)': (r'\texttt{.nbytes} banded', 1),
    'peak dense (teori 2)': ('Peak dense', 2), 'peak band (teori 1)': ('Peak banded', 1),
}
tulis('tabel_kemiringan.tex', [
    f'{nama[i][0]} & {nama[i][1]} & ' + ' & '.join(f'{x:.2f}' for x in k.loc[i]) + r' \\' for i in k.index])

# Lampiran: fungsi inti, disalin apa adanya dari notebook
FUNGSI = ['bfs_jumlah_dikunjungi', 'validasi_T', 'bentuk_B_dense', 'deteksi_bandwidth', 'lu_dense', 'solve_lu',
          'deteksi_bandwidth_T', 'bentuk_B_band', 'lu_band', 'solve_band', 'ukur_waktu']
sel = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
sumber = [''.join(c['source']) for c in sel if c['cell_type'] == 'code']
potongan = []
for f in FUNGSI:
    cocok = [s for s in sumber if f'def {f}(' in s]
    assert len(cocok) == 1, f
    if cocok[0] not in potongan:
        potongan.append(cocok[0])
kode = '\n\n'.join(potongan)
assert kode.isascii(), 'listings butuh ASCII'
(OUT / 'kode_lampiran.py').write_text(kode + '\n', encoding='utf-8')
print('selesai:', sorted(p.name for p in OUT.iterdir()))
