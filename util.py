from sympy import Poly
import numpy as np
import matplotlib.pyplot as plt
import math


def newton(function, x0, max_iter=5000, tol=1e-08, domain='R'):
    """ 
    Newton's method

    Parameters
    ----------
    function: string
        The function that we want to find its root
    x0: float
        The start point
    tol: float  
        the max error we accept
    max_iter: int
        the max iteration we accept
    domain: string
        R for real, C for complex

    Returns
    -------
    int
        number of iteration
    float
        root of func
    """

    func = Poly(function, domain=domain)
    # derivative of func
    der_func = func.diff()

    return _iterate_newton(func, der_func, x0, max_iter, tol)

def _iterate_newton(func, der_func, x0, max_iter=5000, tol=1e-08):
    """
    Iteration process of newton's method
    """

    # convert to a float
    xi = 1.0 * x0
    for i in range(1, max_iter + 1):
        yi = func.eval(xi)

        # iterate
        der_yi = der_func.eval(xi)
        if der_yi == 0:
            # print('Derivative of given function was zero at', xi)
            return i, None
        xj = xi - yi / der_yi

        # close enough
        if math.isclose(xi, xj, rel_tol=0, abs_tol=tol):
            # print('Close enough')
            return i, xj

        xi = xj

    # print('Exceed max iteration')
    return i, xi


def newton_color_map(start, stop, num, function, max_iter=5000, tol=1e-08, domain='R'):
    """
    Compute the color map of newton's method
    """
    
    func = Poly(function, domain=domain)
    # derivative of func
    der_func = func.diff()

    xs = np.linspace(start, stop, num)
    ys = np.zeros(num)
    for i, x in enumerate(xs):
        _, y = _iterate_newton(func, der_func, x, max_iter, tol)
        ys[i] = y

    return ys


def test_plot():
    grid = np.zeros([100, 100, 3],dtype=np.uint8)
    grid[:, 1, 0] = 255
    grid[:, 2, 0] = 255
    grid[:, 3, 0] = 255

    plt.imshow(grid)
    plt.show()

def test_newton():
    # function = input('function: ')
    # x0 = input('start point: ')
    function = "4 + 4 * x - 7 * x**2 + 2 * x**3"
    x0 = 5
    iteration, root = newton(function, x0)
    print("Iteration:", iteration)
    print("Converge to:", f'{root:.15f}')

    # for test purpose
    func = Poly(function, domain='R')
    print('\nAll roots from sympy')
    roots = func.real_roots()
    for r in roots:
        print(r.evalf())

def test_newton_color_map():
    # function = input('function: ')
    # x0 = input('start point: ')
    function = "4 + 2 * x - 7 * x**2 + 2 * x**3"
    x0 = 5
    start = -0.5
    stop = 0.5
    num = 2500

    # TODO: what is the y axis
    ys = newton_color_map(start, stop, num, function, x0)
    ys = np.reshape(ys, (int(math.sqrt(num)),int(math.sqrt(num))))
    # TODO: find an appropriate cmap
    plt.imshow(ys, cmap=plt.cm.RdYlBu)
    plt.show()

    print(ys)

    # for test purpose
    func = Poly(function, domain='R')
    print('\nAll roots from sympy')
    roots = func.real_roots()
    for r in roots:
        print(r.evalf())

if __name__ == "__main__":
    # test_newton()

    test_newton_color_map()
    
    # test_plot()