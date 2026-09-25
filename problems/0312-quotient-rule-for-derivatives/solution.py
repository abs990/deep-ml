import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    def fn_val(coeffs):
        N = len(coeffs) - 1
        value = 0.0
        for coeff in coeffs:
            value += coeff * pow(x, N)
            N -= 1
        return value

    def fn_d(coeffs):
        N = len(coeffs) - 1
        value = 0.0
        for coeff in coeffs[:-1]:
            value += N * coeff * pow(x, N - 1)
            N -= 1
        return value

    return ( fn_d(g_coeffs) * fn_val(h_coeffs) - fn_d(h_coeffs) * fn_val(g_coeffs) ) / (fn_val(h_coeffs) ** 2)