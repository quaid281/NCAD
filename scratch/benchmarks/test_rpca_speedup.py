import time
import numpy as np
import scipy.linalg

def robust_pca_original(
    X: np.ndarray,
    lmbda: float = None,
    max_iter: int = 100,
    tol: float = 1e-7,
):
    X_mat = np.asarray(X, dtype=np.float64)
    orig_shape = X_mat.shape
    if X_mat.ndim > 2:
        X_mat = X_mat.reshape(X_mat.shape[0], -1)

    n, d = X_mat.shape
    if lmbda is None:
        lmbda = 1.0 / np.sqrt(max(n, d))

    norm_two = float(np.linalg.norm(X_mat, 2))
    norm_inf = float(np.max(np.abs(X_mat)) / lmbda)
    dual_norm = max(norm_two, norm_inf)
    if dual_norm < 1e-12:
        dual_norm = 1.0

    Y = X_mat / dual_norm
    L = np.zeros_like(X_mat)
    S = np.zeros_like(X_mat)

    mu = 1.25 / norm_two if norm_two > 1e-12 else 1.25
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = float(np.linalg.norm(X_mat, "fro"))

    for it in range(max_iter):
        temp_s = X_mat - L + (1.0 / mu) * Y
        S = np.sign(temp_s) * np.maximum(np.abs(temp_s) - lmbda / mu, 0.0)

        temp_l = X_mat - S + (1.0 / mu) * Y
        u, s, vh = np.linalg.svd(temp_l, full_matrices=False)
        s_th = np.maximum(s - 1.0 / mu, 0.0)
        rank = int(np.sum(s_th > 0))
        if rank > 0:
            L = (u[:, :rank] * s_th[:rank]) @ vh[:rank, :]
        else:
            L = np.zeros_like(X_mat)

        Z = X_mat - L - S
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        err = float(np.linalg.norm(Z, "fro")) / (d_norm + 1e-12)
        if err < tol:
            break

    if len(orig_shape) > 2:
        L = L.reshape(orig_shape)
        S = S.reshape(orig_shape)

    return L.astype(np.float32), S.astype(np.float32)

def robust_pca_fast(
    X: np.ndarray,
    lmbda: float = None,
    max_iter: int = 50,
    tol: float = 1e-6,
):
    X_mat = np.asarray(X, dtype=np.float32)
    orig_shape = X_mat.shape
    if X_mat.ndim > 2:
        X_mat = X_mat.reshape(X_mat.shape[0], -1)

    n, d = X_mat.shape
    if lmbda is None:
        lmbda = 1.0 / np.sqrt(max(n, d))

    # Fast norm approximations in float32
    norm_two = float(scipy.linalg.svdvals(X_mat)[0]) if min(n, d) <= 500 else float(np.linalg.norm(X_mat, 2))
    norm_inf = float(np.max(np.abs(X_mat)) / lmbda)
    dual_norm = max(norm_two, norm_inf)
    if dual_norm < 1e-12:
        dual_norm = 1.0

    Y = X_mat / np.float32(dual_norm)
    L = np.zeros_like(X_mat)
    S = np.zeros_like(X_mat)

    mu = np.float32(1.25 / norm_two if norm_two > 1e-12 else 1.25)
    mu_bar = np.float32(mu * 1e7)
    rho = np.float32(1.5)
    d_norm = float(np.linalg.norm(X_mat, "fro"))
    inv_d_norm = 1.0 / (d_norm + 1e-12)

    for _ in range(max_iter):
        inv_mu = 1.0 / mu
        temp_s = X_mat - L + inv_mu * Y
        S = np.sign(temp_s) * np.maximum(np.abs(temp_s) - (lmbda * inv_mu), 0.0)

        temp_l = X_mat - S + inv_mu * Y
        # Fast LAPACK gesdd SVD in float32
        u, s, vh = scipy.linalg.svd(temp_l, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver="gesdd")
        s_th = np.maximum(s - inv_mu, 0.0)
        rank = int(np.sum(s_th > 0))
        if rank > 0:
            L = (u[:, :rank] * s_th[:rank]) @ vh[:rank, :]
        else:
            L.fill(0.0)

        Z = X_mat - L - S
        Y += mu * Z
        mu = min(mu * rho, mu_bar)

        err = float(np.linalg.norm(Z, "fro")) * inv_d_norm
        if err < tol:
            break

    if len(orig_shape) > 2:
        L = L.reshape(orig_shape)
        S = S.reshape(orig_shape)

    return L.astype(np.float32), S.astype(np.float32)

np.random.seed(42)
# Low rank (rank 5) + sparse corruptions (shape 500 x 200)
L_true = np.random.randn(500, 5) @ np.random.randn(5, 200)
S_true = np.zeros((500, 200))
mask = np.random.rand(500, 200) < 0.05
S_true[mask] = np.random.randn(np.sum(mask)) * 5.0
X_test = L_true + S_true

t0 = time.time()
L_orig, S_orig = robust_pca_original(X_test)
t_orig = time.time() - t0

t0 = time.time()
L_fast, S_fast = robust_pca_fast(X_test)
t_fast = time.time() - t0

diff = np.max(np.abs(L_orig - L_fast))
print(f"Original RPCA time: {t_orig:.4f}s")
print(f"Fast RPCA time:     {t_fast:.4f}s ({t_orig / t_fast:.2f}x speedup)")
print(f"Max diff in low-rank L: {diff:.2e}")
assert diff < 1e-3, "RPCA difference exceeds threshold"
print("Fast RPCA verified successfully!")
