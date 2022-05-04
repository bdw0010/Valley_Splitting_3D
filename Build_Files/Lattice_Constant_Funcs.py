"""
    Lattice Constant functions
"""

import numpy as np

def calc_relaxed_lat_constant(x):
    a_Si = 5.430
    a_Ge = 5.6563
    a_relaxed = (1 - x) * a_Si + x * a_Ge
    return a_relaxed

def calc_m_perp_lat_constant(a_par,x_m):
    ### Calculate the perpendicular lattice constant
    ### if the entire quantum well had x_m as its Ge fraction
    c12_c11 = (63.9 - 15.6 * x_m)/ (165.8 - 37.3*x_m)
    a_m_relaxed = calc_relaxed_lat_constant(x_m)
    a_perp_m = a_m_relaxed * (1. - 2*c12_c11 * (a_par/a_m_relaxed - 1.))
    return a_perp_m
