import numpy as np

def rank_k_approx(A, k):
  U, s, Vt = np.linalg.svd(A, full_matrices=False)
  return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

def storage(m, n, k):
  return k * (m + n + 1)

def compression_ratio(m, n, k):
  return (m * n) / storage(m, n, k)

def rel_error(A, B):
  return np.linalg.norm(A - B) / np.linalg.norm(A)

def predicted_error(s, k):
  return np.sqrt(np.sum(s[k:]**2) / np.sum(s**2))
