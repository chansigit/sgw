import numpy as np


def sample_pairs_from_plan(
    T: np.ndarray, M: int
) -> list[tuple[int, int]]:
    """Sample M (row, col) pairs from transport plan T, weighted by mass.

    Uses the Gumbel-max trick for vectorized categorical sampling,
    avoiding a Python for-loop over M pairs.

    Parameters
    ----------
    T : ndarray of shape (N, K), non-negative
    M : number of pairs to sample

    Returns
    -------
    pairs : list of (row_index, col_index) tuples
    """
    N, K = T.shape
    p_rows = T.sum(axis=1)
    total = p_rows.sum()
    if total < 1e-9:
        rows = np.random.randint(0, N, size=M)
        cols = np.random.randint(0, K, size=M)
        return list(zip(rows, cols))

    p_rows = p_rows / total
    sampled_rows = np.random.choice(N, size=M, p=p_rows)

    # Gumbel-max trick: vectorized categorical sampling over columns
    row_slices = T[sampled_rows]  # (M, K)
    row_sums = row_slices.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-30)
    log_probs = np.log(row_slices / row_sums + 1e-30)
    gumbel_noise = -np.log(-np.log(np.random.uniform(size=(M, K)) + 1e-30) + 1e-30)
    sampled_cols = np.argmax(log_probs + gumbel_noise, axis=1)

    return list(zip(sampled_rows, sampled_cols))
