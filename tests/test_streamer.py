import numpy as np

from labnewt import StencilD2Q9, Streamer


def test_streamer_with_d2q9_stencil():
    s = StencilD2Q9()
    streamer = Streamer()
    i, j, k = np.indices((9, 3, 3))
    f = 1.0 * i + 0.1 * j + 0.01 * k
    f = streamer.stream(f, s)

    # No change for q=0 (e = [0, 0])
    assert np.allclose(
        f[0, :, :],
        np.array([[0.0, 0.01, 0.02], [0.1, 0.11, 0.12], [0.2, 0.21, 0.22]]),
        atol=1.0e-12,
    )

    # Rightward shift for q=1 (e = [1, 0])
    assert np.allclose(
        f[1, :, :],
        np.array([[1.2, 1.21, 1.22], [1.0, 1.01, 1.02], [1.1, 1.11, 1.12]]),
        atol=1.0e-12,
    )

    # Leftward shift for q=2 (e = [-1, 0])
    assert np.allclose(
        f[2, :, :],
        np.array([[2.1, 2.11, 2.12], [2.2, 2.21, 2.22], [2.0, 2.01, 2.02]]),
        atol=1.0e-12,
    )

    # Upwards shift for q=3 (e = [0, 1])
    assert np.allclose(
        f[3, :, :],
        np.array([[3.02, 3.00, 3.01], [3.12, 3.10, 3.11], [3.22, 3.20, 3.21]]),
        atol=1.0e-12,
    )

    # Downwards shift for q=4 (e = [0, -1])
    assert np.allclose(
        f[4, :, :],
        np.array([[4.01, 4.02, 4.00], [4.11, 4.12, 4.10], [4.21, 4.22, 4.20]]),
        atol=1.0e-12,
    )

    # Up and right shift for q=5 (e = [1, 1])
    assert np.allclose(
        f[5, :, :],
        np.array([[5.22, 5.20, 5.21], [5.02, 5.00, 5.01], [5.12, 5.10, 5.11]]),
        atol=1.0e-12,
    )

    # Down and left shift for q=6 (e = [-1, -1])
    assert np.allclose(
        f[6, :, :],
        np.array([[6.11, 6.12, 6.10], [6.21, 6.22, 6.20], [6.01, 6.02, 6.00]]),
        atol=1.0e-12,
    )

    # Up and left shift for q=7 (e = [-1, 1])
    assert np.allclose(
        f[7, :, :],
        np.array([[7.12, 7.10, 7.11], [7.22, 7.20, 7.21], [7.02, 7.00, 7.01]]),
        atol=1.0e-12,
    )

    # Down and right shift for q=8 (e = [1, -1])
    assert np.allclose(
        f[8, :, :],
        np.array([[8.21, 8.22, 8.20], [8.01, 8.02, 8.00], [8.11, 8.12, 8.10]]),
        atol=1.0e-12,
    )
