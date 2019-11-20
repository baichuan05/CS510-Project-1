from sympy import Poly
import math

def newton(function, x0, tol=1e-08, max_iter=500, domain='R'):
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
    # convert to a float
    xi = 1.0 * x0

    for i in range(1, max_iter + 1):
        yi = func.eval(xi)

        # iterate
        der_yi = der_func.eval(xi)
        if der_yi == 0:
            print('Derivative of given function was zero at', xi)
            return i, None
        xj = xi - yi / der_yi

        # close enough
        if math.isclose(xi, xj, rel_tol=0, abs_tol=tol):
            print('Close enough')
            return i, xj

        xi = xj

    print('Exceed max iteration')
    return i, xi

if __name__ == "__main__":
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