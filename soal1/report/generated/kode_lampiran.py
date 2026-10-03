def bfs_jumlah_dikunjungi(T: np.ndarray, balik: bool = False) -> int:
    N = T.shape[0]
    dikunjungi = np.zeros(N, dtype=bool)
    dikunjungi[0] = True
    antrean = deque([0])
    while antrean:
        u = antrean.popleft()
        # graf asli: tetangga dari baris u; graf dibalik: dari kolom u
        if balik:
            tetangga = np.nonzero(T[:, u] > 0)[0]
        else:
            tetangga = np.nonzero(T[u, :] > 0)[0]
        for v in tetangga:
            if not dikunjungi[v]:
                dikunjungi[v] = True
                antrean.append(v)
    return int(dikunjungi.sum())

def validasi_T(T: np.ndarray, N: int, tol: float = TOL_BARIS) -> dict:
    hasil = {'N': N}
    hasil['dimensi_ok'] = T.shape == (N, N)
    hasil['hingga'] = bool(np.all(np.isfinite(T)))
    if not (hasil['dimensi_ok'] and hasil['hingga']):
        hasil['valid'] = False
        return hasil

    hasil['nonnegatif'] = bool(T.min() >= 0)
    hasil['galat_baris_maks'] = float(np.max(np.abs(T.sum(axis=1) - 1)))
    hasil['baris_ok'] = hasil['galat_baris_maks'] <= tol
    # terhubung kuat, semua halte tercapai dari halte 1 DAN halte 1 tercapai dari semua halte
    hasil['irreducible'] = (bfs_jumlah_dikunjungi(T) == N
                            and bfs_jumlah_dikunjungi(T, balik=True) == N)
    hasil['aperiodik'] = bool(np.any(np.diag(T) > 0))
    hasil['min_diag'] = float(np.diag(T).min())
    # peluang pindah terkecil yang masih positif; sangat kecil = rantai "hampir terputus"
    luar_diag = T - np.diag(np.diag(T))
    hasil['min_taknol_luar_diag'] = float(luar_diag[luar_diag > 0].min()) if np.any(luar_diag > 0) else 0.0

    nnz = np.count_nonzero(T, axis=1)
    hasil['nnz_per_baris'] = f'{nnz.min()}-{nnz.max()}'
    i, j = np.nonzero(T)
    hasil['offset_taknol'] = str(sorted(set((j - i).tolist()))) # nilai j - i dari elemen tak-nol

    hasil['valid'] = hasil['nonnegatif'] and hasil['baris_ok'] and hasil['irreducible']
    return hasil

def bentuk_B_dense(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = T.shape[0]
    B = np.eye(N) - T.T          # A = I - T^T
    B[0, :] = 0.0                # baris 1 diganti [1 0 ... 0]
    B[0, 0] = 1.0
    b = np.zeros(N)
    b[0] = 1.0
    return B, b

def deteksi_bandwidth(M: np.ndarray) -> tuple[int, int]:
    '''Menentukan bandwidth dari posisi elemen tak-nol.

    Input : M (N x N).
    Output: p = maks(i - j) (lower), q = maks(j - i) (upper).
    '''
    i, j = np.nonzero(M)
    p = max(0, int(np.max(i - j)))
    q = max(0, int(np.max(j - i)))
    return p, q

def lu_dense(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    LU = B.astype(float).copy()
    N = LU.shape[0]
    perm = np.arange(N)
    for k in range(N - 1):
        # pivot: elemen dengan nilai mutlak terbesar di kolom k, baris k..N-1
        r = k + int(np.argmax(np.abs(LU[k:, k])))
        if LU[r, k] == 0.0:
            raise ValueError(f'B singular: kolom {k} tidak punya pivot tak-nol')
        if r != k:
            LU[[k, r], :] = LU[[r, k], :] # tukar seluruh baris, termasuk multiplier lama
            perm[[k, r]] = perm[[r, k]]
        # multiplier l_ik disimpan di tempat elemen yang dieliminasi
        LU[k + 1:, k] /= LU[k, k]
        # baris i dikurangi l_ik * baris k (kolom k+1 ke kanan)
        LU[k + 1:, k + 1:] -= np.outer(LU[k + 1:, k], LU[k, k + 1:])
    if LU[N - 1, N - 1] == 0.0:
        raise ValueError('B singular: pivot terakhir nol')
    return LU, perm

def solve_lu(LU: np.ndarray, perm: np.ndarray, b: np.ndarray) -> np.ndarray:
    N = LU.shape[0]
    y = b[perm].astype(float) # Pb
    for i in range(1, N): # substitusi maju Ly = Pb (diagonal L = 1)
        y[i] -= np.dot(LU[i, :i], y[:i])
    z = np.zeros(N)
    for i in range(N - 1, -1, -1): # substitusi mundur Uz = y
        z[i] = (y[i] - np.dot(LU[i, i + 1:], z[i + 1:])) / LU[i, i]
    return z

def deteksi_bandwidth_T(T: np.ndarray) -> tuple[int, int]:
    i, j = np.nonzero(T)
    pakai = j >= 1
    p = max(0, int(np.max(j[pakai] - i[pakai])))
    q = max(0, int(np.max(i[pakai] - j[pakai])))
    return p, q

def bentuk_B_band(T: np.ndarray, p: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    N = T.shape[0]
    d = p + q # baris AB tempat diagonal utama
    AB = np.zeros((2 * p + q + 1, N))
    AB[d, 0] = 1.0 # baris 1 B = [1 0 ... 0]
    for i in range(1, N):
        for j in range(max(0, i - p), min(N - 1, i + q) + 1):
            AB[d + i - j, j] = (1.0 if i == j else 0.0) - T[j, i]
    b = np.zeros(N)
    b[0] = 1.0
    return AB, b

def lu_band(AB: np.ndarray, p: int, q: int) -> tuple[np.ndarray, np.ndarray]:
    AB = AB.copy()
    d = p + q
    N = AB.shape[1]
    piv = np.zeros(N, dtype=int)
    for k in range(N):
        i_maks = min(N - 1, k + p) 
        j_maks = min(N - 1, k + p + q) 
        r = k + int(np.argmax(np.abs(AB[d:d + i_maks - k + 1, k])))
        if AB[d + r - k, k] == 0.0:
            raise ValueError(f'B singular: kolom {k} tidak punya pivot tak-nol')
        piv[k] = r
        js = np.arange(k, j_maks + 1)
        if r != k:
            sementara = AB[d + k - js, js]
            AB[d + k - js, js] = AB[d + r - js, js]
            AB[d + r - js, js] = sementara
        js = js[1:]
        for i in range(k + 1, i_maks + 1):
            AB[d + i - k, k] /= AB[d, k]
            AB[d + i - js, js] -= AB[d + i - k, k] * AB[d + k - js, js]
    return AB, piv

def solve_band(AB: np.ndarray, piv: np.ndarray, p: int, q: int, b: np.ndarray) -> np.ndarray:
    d = p + q
    N = AB.shape[1]
    y = b.astype(float).copy()
    for k in range(N):
        r = piv[k]
        if r != k:
            y[k], y[r] = y[r], y[k]
        for i in range(k + 1, min(N - 1, k + p) + 1):
            y[i] -= AB[d + i - k, k] * y[k]
    z = np.zeros(N)
    for i in range(N - 1, -1, -1): 
        js = np.arange(i + 1, min(N - 1, i + p + q) + 1)
        z[i] = (y[i] - np.dot(AB[d + i - js, js], z[js])) / AB[d, i]
    return z

def ukur_waktu(fungsi, ulang: int = ULANG) -> float:
    fungsi()
    waktu = []
    for _ in range(ulang):
        t0 = time.perf_counter()
        fungsi()
        waktu.append(time.perf_counter() - t0)
    return float(np.min(waktu))


def ukur_peak(fungsi):
    tracemalloc.start()
    hasil = fungsi()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak, hasil
