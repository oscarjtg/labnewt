import numpy as np

from labnewt import StencilD2Q9, Streamer


def test_streamer_with_d2q9_stencil():
    s = StencilD2Q9()
    streamer = Streamer()
    i, j, k = np.indices((9, 3, 3))
    fo = 1.0 * i + 0.1 * j + 0.01 * k
    fo0 = np.copy(fo)
    fi = np.empty_like(fo)
    streamer._stream(fi, fo, s)

    # fo should not be changed
    assert np.allclose(fo, fo0, atol=1.0e-12)

    # fi should be as follows:
    # No change for q=0 (e = [0, 0])
    assert np.allclose(
        fi[0, :, :],
        np.array([[0.0, 0.01, 0.02], [0.1, 0.11, 0.12], [0.2, 0.21, 0.22]]),
        atol=1.0e-12,
    )

    # Rightward shift for q=1 (e = [1, 0])
    assert np.allclose(
        fi[1, :, :],
        np.array([[1.02, 1.00, 1.01], [1.12, 1.10, 1.11], [1.22, 1.20, 1.21]]),
        atol=1.0e-12,
    )

    # Leftward shift for q=2 (e = [-1, 0])
    assert np.allclose(
        fi[2, :, :],
        np.array([[2.01, 2.02, 2.00], [2.11, 2.12, 2.10], [2.21, 2.22, 2.20]]),
        atol=1.0e-12,
    )

    # Upwards shift for q=3 (e = [0, 1])
    assert np.allclose(
        fi[3, :, :],
        np.array([[3.2, 3.21, 3.22], [3.0, 3.01, 3.02], [3.1, 3.11, 3.12]]),
        atol=1.0e-12,
    )

    # Downwards shift for q=4 (e = [0, -1])
    assert np.allclose(
        fi[4, :, :],
        np.array([[4.1, 4.11, 4.12], [4.2, 4.21, 4.22], [4.0, 4.01, 4.02]]),
        atol=1.0e-12,
    )

    # Up and right shift for q=5 (e = [1, 1])
    assert np.allclose(
        fi[5, :, :],
        np.array([[5.22, 5.20, 5.21], [5.02, 5.00, 5.01], [5.12, 5.10, 5.11]]),
        atol=1.0e-12,
    )

    # Down and left shift for q=6 (e = [-1, -1])
    assert np.allclose(
        fi[6, :, :],
        np.array([[6.11, 6.12, 6.10], [6.21, 6.22, 6.20], [6.01, 6.02, 6.00]]),
        atol=1.0e-12,
    )

    # Up and left shift for q=7 (e = [-1, 1])
    assert np.allclose(
        fi[7, :, :],
        np.array([[7.21, 7.22, 7.20], [7.01, 7.02, 7.00], [7.11, 7.12, 7.10]]),
        atol=1.0e-12,
    )

    # Down and right shift for q=8 (e = [1, -1])
    assert np.allclose(
        fi[8, :, :],
        np.array([[8.12, 8.10, 8.11], [8.22, 8.20, 8.21], [8.02, 8.00, 8.01]]),
        atol=1.0e-12,
    )
