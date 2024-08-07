from numpy import power, sqrt

def hilln(x, n):
    """
    Given an array x and a power n, return a Hill function of nth power.
    """
    return power(x, n)/(1+power(x, n))

def logistic(x, x_end):
    """
    Given an array x, and a float ending density x_end, return logistic function.
    """
    return x*(1-(x/x_end))

def sqrtabs(x, arg):
    """
    Given an array x, return sqrt(abs(x))
    """
    return sqrt(abs(x))