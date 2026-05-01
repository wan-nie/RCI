import numpy as np
import matplotlib.pyplot as plt


def sample_block_coords(rows, cols, k, rng, upper_triangle_only=False):
    R = len(rows)
    C = len(cols)
    # rng = np.random.default_rng(seed)

    if not upper_triangle_only:
        total = R * C
        if k > total:
            raise ValueError(f"k={k} exceeds block size {total} (= {R}*{C})")
    
        flat = rng.choice(total, size=k, replace=False)
        r_local = flat // C
        c_local = flat % C 

        coords = np.stack((rows[r_local], cols[c_local]), axis=1)
    
    else:
        assert R == C and np.all(rows == cols)

        m = R
        cap = m * (m - 1) // 2
        if k > cap:
            raise ValueError(f"k={k} exceeds upper-triangle capacity {cap}")

        flat_ut = rng.choice(cap, size=k, replace=False)
        def t_to_ij(t_arr, m):
            """
            Docstring for t_to_ij
            S(i) = \\sum_{k=0}^{i-1} (m-k-1)  # Sum of items before row i, i in [0, m-1]
                = i(2m -i - 1)/2
            t = S(i) --> i = floor(m - 0.5 - sqrt( (m - 0.5)^2 - 2t ))
            """
            t = np.asarray(t_arr, dtype=np.int64)

            #
            a = m - 0.5
            i_float = a - np.sqrt(a**2 - 2*t)
            i = np.floor(i_float).astype(np.int64)
            assert np.all(i >= 0) and np.all(i < m-1)

            #
            Si = i * (2*m - i - 1) // 2
            Sip1 = (i + 1) * (2*m - (i+1) - 1) // 2
            assert np.all(t >= Si) and np.all(t < Sip1)

            #
            off = t - Si
            j = (i + 1 + off).astype(np.int64)

            #
            # print(i_float, i, Si, Sip1, off)

            return i, j
        
        r_local, c_local = t_to_ij(flat_ut, m)
        coords = np.stack((rows[r_local], cols[c_local]), axis=1)

    return coords


def plot_heatmap(matrix: np.ndarray, out_path: str, title: str = "weights", cmap: str = "magma"):
    if '.csv' in out_path:
        np.savetxt(out_path, matrix, delimiter=",", fmt="%.3e")
    else:
        B = matrix.shape[0]
        plt.figure(figsize=(max(5, B * 0.3), max(4, B * 0.3)))
        im = plt.imshow(matrix, interpolation="nearest", cmap=cmap)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(title)
        plt.xlabel("bin a")
        plt.ylabel("bin b")
        plt.xticks(range(B))
        plt.yticks(range(B))
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

