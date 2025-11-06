import matplotlib.pyplot as plt

from labnewt import Model


def func(x, y, a, b):
    return a * x + b * y


if __name__ == "__main__":
    nx = 10
    ny = 10
    dx = 0.1
    dt = 0.1
    nu = 0.01
    model = Model(nx, ny, dx, dt, nu)

    model.set_r(func, 0.0, -1.0)
    model.set_u(func, 1.0, 0.0)
    model.set_v(func, 0.0, 1.0)
    model.plot_fields()
    plt.show()
