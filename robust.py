import matplotlib.pyplot as plt
import numpy as np
import sympy
import cmath
import math
from matplotlib.colors import LinearSegmentedColormap

colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
n_bins = 6  # Discretizes the interpolation into bins
cmap_name = 'my_cm'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


def _iterate_robust_step(func_list, func, deriv_func, xi, tol):
    fi = func(xi)
    der_yi = deriv_func(xi)

    k = 1
    tmp_func = func_list[1]
    A = abs(fi)
    while abs(der_yi) < tol and k + 1 < len(func_list):
        k += 1
        tmp_func = func_list[k]
        der_yi = tmp_func(xi)

    uk = fi * der_yi.conjugate() / math.factorial(k)
    A = max(A, abs(der_yi) / math.factorial(k))
    j = k
    while j + 1 < len(func_list):
        j += 1
        tmp_func = func_list[j]
        der_yi = tmp_func(xi)
        A = max(A, abs(der_yi) / math.factorial(j))

    ukk = uk ** (k - 1)
    gamma = 2 * ukk.real
    delta = -2 * ukk.imag
    if abs(gamma) >= abs(delta):
        ck = abs(gamma)
        if gamma < 0:
            theta = 0
        else:
            theta = cmath.pi / k
    else:
        ck = abs(delta)
        if delta < 0:
            theta = cmath.pi / (2 * k)
        else:
            theta = 3 * cmath.pi / (2 * k)
    Ck = ck * (abs(uk) ** (2 - k)) / (6 * A * A)
    xj = xi + Ck * uk * cmath.exp(complex(0, 1) * theta) / abs(uk) / 3.0
    return xj


def _iterate_robust(func_list, func, deriv_func, x0, max_iter, tol):
    """
    Iteration process of robust newton's method

    Parameters
    ----------
    func: function
        the function
    deriv_func: function
        the derivative of the function
    """
    tol2 = 1e-09
    xi = x0
    # print(func(xi), deriv_func(xi), func(xi) * deriv_func(xi), tol2)
    for i in range(1, max_iter + 1):
        if abs(func(xi)) < tol2 or abs(deriv_func(xi)) < tol2:
            return i, xi
        xj = _iterate_robust_step(func_list, func, deriv_func, xi, tol)
        xi = xj
    xi = -100
    return i, xi


def robust_color_map(function, interval, num, max_iter=1000, tol=2e-03, decimals=3):
    """
    Compute the color map of robust newton's method

    Parameters
    ----------
    interval: tuple with size of 4
        define the range of real and complex parts
        (real_min, real_max, complex_min, complex_max)
    num: tuple with size of 2
        define the number of points
        (num_real, num_complex)

    Returns
    -------
    tuple
        all roots found in the interval
    2D numpy array
        class of a point
    """
    x = sympy.Symbol('x')
    func = eval("lambda x: " + function)
    deriv_func = eval("lambda x: " + str(sympy.diff(function, x)))
    func_list = [eval("lambda x: " + str(function))]
    while function:
        function = sympy.diff(function, x)
        func_list.append(eval("lambda x: " + str(function)))

    root_count = 0
    roots = []
    color_map = np.zeros((num[0], num[1]))

    resolution = (interval[1] - interval[0]) / num[0]
    r = interval[0]
    for i in range(num[0]):
        c = interval[2]
        for j in range(num[0]):
            x0 = np.round(r + c*1j, decimals)
            root = np.round(_iterate_robust(func_list, func, deriv_func, x0, max_iter, tol)[1], decimals)
            new_root = True
            for k in range(len(roots)):
                if abs(root - roots[k]) < 0.01:
                    color_map[i, j] = k
                    new_root = False
            if new_root is True:
                root_count += 1
                roots.append(root)
                color_map[i, j] = root_count - 1

            c += resolution
        r = np.round(r + resolution, decimals)
        if i % 10 == 0:
            print(i)
        if len(roots) > 5:
            print(roots)
    # x0 = 0.002+0.002j
    # root = np.round(_iterate_robust(func_list, func, deriv_func, x0, max_iter, tol)[1], decimals)
    # print(root)
    return tuple(roots), color_map


def test_robust_color_map():
    user_input = "x**4 - 1"
    interval = (-3, 3, -3, 3)
    num = (6000, 6000)

    roots, color_map = robust_color_map(user_input, interval, num)
    print(roots)

    # for test purpose
    func = sympy.Poly(user_input)
    print('\nAll roots from sympy')
    roots = func.all_roots()
    print(roots)

    # TODO: find a good cmap
    plt.axis('off')
    plt.imshow(color_map.T, cmap=cm, extent=interval)
    plt.show()


if __name__ == "__main__":
    test_robust_color_map()
