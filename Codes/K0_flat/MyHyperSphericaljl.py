import torch as tc
tc.set_default_tensor_type(tc.DoubleTensor)

import scipy as sp

###############################################
# K = 0

def TurningPoint(l):
    return tc.sqrt(l * (l+1))

# Definitions for sink and tank functions

def sink(x):
    return x

def tank(x):
    return x

def dsink(x):
    return tc.ones_like(x)

def dtank(x):
    return tc.ones_like(x)

# Definitions for l=0 and l=1 modes

def Phi0nu(x):
    return tc.sin(x)/ sink(x)

def Phi1nu(x):
    return Phi0nu(x) * (1 / tank(x) - 1 / tc.tan(x))

# Recurrence coefficients

def alphal(l, x):
    return (2*l + 1) / tank(x)

def betal(l):
    return tc.tensor([-1], dtype=tc.float64)

def ratio_list_modified(x, lmax):

    res = tc.zeros([lmax, len(x)])
    l_large = 2 * lmax

    temp = - tank(x) / (2 * l_large + 1)
    for ll in tc.arange(l_large - 1, lmax - 1, -1):
        temp = betal(ll) / (alphal(ll, x) + temp)
        
    res[-1] = temp # which corresponds to y_{lmax}/y_{lmax-1}
    llist = tc.arange(lmax - 2, 0, -1)
    for li in llist:
        res[li] = betal(li + 1) / (alphal(li + 1, x) + res[li+1])

    return res

###############################################

def Recurrence(x, lmax = 2000):
    '''
        Input:  k           float (=nu in non-flat case)
                x           (N) torch tensor

        Output: j_l         (lmax * N) torch tensor
    '''

    phi = tc.zeros([lmax, len(x)])

    phi[0] = Phi0nu(x)
    phi[1] = Phi1nu(x)

    llist = tc.arange(2, lmax, dtype=tc.int32)
    rlist = ratio_list_modified(x, lmax)

    for li in llist:
        phi[li] = tc.where(x > TurningPoint(li),  # turning point condition
                            alphal(li-1, x) * phi[li-1] + betal(li-1) * phi[li-2],  # forward recurrence
                            -rlist[li - 1] * phi[li - 1]) # backward recurrence

    return phi #, dphi

def Recurrence_specified(x, lmax = 2000):
    '''
        Input:  k           float (=nu in non-flat case)
                x           (N) torch tensor

        Output: j_l         (lmax * N) torch tensor
    '''
    
    phi = tc.empty([lmax, len(x)])

    phi[0] = Phi0nu(x)
    phi[1] = Phi1nu(x)

    llist = tc.arange(2, lmax, dtype=tc.int32)
    rlist = ratio_list_modified(x, lmax)

    for li in llist:
        phi[li] = tc.where(x > TurningPoint(li),  # turning point condition
                            alphal(li-1, x) * phi[li-1] + betal(li-1) * phi[li-2],  # forward recurrence
                            -rlist[li - 1] * phi[li - 1]) # backward recurrence

    phi_at_0 = tc.zeros([lmax, 1])
    phi_at_0[0] = tc.tensor([1.])
    phi_tot = tc.hstack([phi, phi_at_0])

    return phi_tot #, dphi

###############################################

def sp_jl(k, x, lmax = 2000):
    '''
        Input:  k           float
                x           (N) torch tensor

        Output: j_l^nu    (lmax * N) torch tensor
    '''

    x_matrix = k * x * tc.ones([lmax, len(x)])
    l_matrix = tc.arange(lmax, dtype=tc.int32).reshape([lmax,1])

    phi = sp.special.spherical_jn(l_matrix, x_matrix)

    return phi #, dphi