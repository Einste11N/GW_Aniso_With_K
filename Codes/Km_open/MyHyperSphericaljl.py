import torch as tc
tc.set_default_tensor_type(tc.DoubleTensor)

# K = -1

def TurningPoint(nu, l):
    return tc.arcsinh(tc.sqrt(l*(l+1) / nu**2))

# Definitions for sink and tank functions

def sink(x):
    return tc.sinh(x)

def tank(x):
    return tc.tanh(x)

def dsink(x):
    return tc.cosh(x)

def dtank(x):
    return 1 / tc.cosh(x)**2

# Definitions for l=0 and l=1 modes

def Phi0nu(nu, x):
    return tc.sin(nu * x)/ nu / sink(x)

def Phi1nu(nu, x):
    return Phi0nu(nu, x) * (1 / tank(x) - nu / tc.tan(nu * x)) / tc.sqrt(nu**2 + 1)

def dPhi0nu(nu, x):
    return (nu * tc.cos(nu * x) * sink(x) - tc.sin(nu * x) * dsink(x)) / nu / sink(x)**2

# def dPhiByPhi_recurrence(nu, l, ratio, x):
#     return l / tank(x) + tc.sqrt(nu**2 + (l+1)**2) * ratio

# Recurrence coefficients

def alphal(nu, l, x):
    return (2*l + 1) / tank(x) / tc.sqrt(nu**2 + (l+1)**2)

def betal(nu, l):
    return -tc.sqrt(nu**2 + l**2) / tc.sqrt(nu**2 + (l+1)**2)

def ratio_list_modified(nu, lmax, x):
    '''
        ratio_l := -y_{l+1} / y_l = beta_{l+1}/(alpha_{l+1} + ...)
                0<=l<=lmax-1
    '''
    res = tc.zeros([lmax, len(x)])

    l_large = 2 * lmax
    
    temp = - tc.sqrt(nu**2 + (l_large + 1)**2) * tank(x) / (2 * l_large + 1)
    for ll in tc.arange(l_large - 1, lmax - 1, -1):
        temp = betal(nu, ll) / (alphal(nu, ll, x) + temp)
        
    res[-1] = temp # which corresponds to y_{lmax}/y_{lmax-1}
    llist = tc.arange(lmax - 2, 0, -1)
    for li in llist:
        res[li] = betal(nu, li + 1) / (alphal(nu, li + 1, x) + res[li+1])

    return res

# Computation

def Recurrence(nu, x, lmax = 2000):
    '''
        Input:  nu          1*1 torch.float64
                x           (N) torch tensor

        Output: Phi_l^nu    (lmax * N) torch tensor with 0<=l
    '''

    phi = tc.zeros([lmax, len(x)])
    # dphi = tc.zeros([lmax, len(x)])

    phi[0] = Phi0nu(nu, x)
    phi[1] = Phi1nu(nu, x)

    llist = tc.arange(2, lmax)
    rlist = ratio_list_modified(nu, lmax, x)

    # dphi[0] = dPhi0nu(nu, x)
    # dphi[1] = dPhiByPhi_recurrence(nu, 1, rlist[0], x) * phi[1]

    for li in llist:
        phi[li] = tc.where(x > TurningPoint(nu, li),  # turning point condition
                            alphal(nu, li-1, x) * phi[li-1] + betal(nu, li-1) * phi[li-2],  # forward recurrence
                            -rlist[li - 1] * phi[li - 1]) # backward recurrence
        
        # dphi[li] = dPhiByPhi_recurrence(nu, li, rlist[li], x) * phi[li]

    return phi#, dphi

def Recurrence_specified(nu, x, lmax = 2000):
    '''
        Input:  nu          1*1 torch.float64
                x           (N) torch tensor

        Output: Phi_l^nu    (lmax * N) torch tensor with 0<=l
    '''

    phi = tc.zeros([lmax, len(x)])

    phi[0] = Phi0nu(nu, x)
    phi[1] = Phi1nu(nu, x)

    llist = tc.arange(2, lmax)
    rlist = ratio_list_modified(nu, lmax, x)

    for li in llist:
        phi[li] = tc.where(x > TurningPoint(nu, li),  # turning point condition
                            alphal(nu, li-1, x) * phi[li-1] + betal(nu, li-1) * phi[li-2],  # forward recurrence
                            -rlist[li - 1] * phi[li - 1]) # backward recurrence
    
    phi_at_0 = tc.zeros([lmax, 1])
    phi_at_0[0] = tc.tensor([1.])
    phi_tot = tc.hstack([phi, phi_at_0])

    return phi_tot

################################################################################
################################################################################

def ratio_list_SW(nu, lmax, x):
    '''
        ratio_l := -y_{l+1} / y_l = beta_{l+1}/(alpha_{l+1} + ...)
                0<=l<=lmax-1
    '''
    res = tc.zeros([lmax, len(nu)])

    l_large = 2 * lmax
    
    temp = - tc.sqrt(nu**2 + (l_large + 1)**2) * tank(x) / (2 * l_large + 1)
    for ll in tc.arange(l_large - 1, lmax - 1, -1):
        temp = betal(nu, ll) / (alphal(nu, ll, x) + temp)
        
    res[-1] = temp # which corresponds to y_{lmax}/y_{lmax-1}
    llist = tc.arange(lmax - 2, 0, -1)
    for li in llist:
        res[li] = betal(nu, li + 1) / (alphal(nu, li + 1, x) + res[li+1])

    return res

def Recurrence_SW(nu, x, lmax = 2000):
    '''
        Input:  nu          (N) torch tensor
                x           1*1 torch.float64

        Output: Phi_l^nu    (lmax * N) torch tensor with 0<=l
    '''

    phi = tc.zeros([lmax, len(nu)])
    # dphi = tc.zeros([lmax, len(x)])

    phi[0] = Phi0nu(nu, x)
    phi[1] = Phi1nu(nu, x)

    llist = tc.arange(2, lmax)
    rlist = ratio_list_SW(nu, lmax, x)

    # dphi[0] = dPhi0nu(nu, x)
    # dphi[1] = dPhiByPhi_recurrence(nu, 1, rlist[0], x) * phi[1]

    for li in llist:
        phi[li] = tc.where(x > TurningPoint(nu, li),  # turning point condition
                            alphal(nu, li-1, x) * phi[li-1] + betal(nu, li-1) * phi[li-2],  # forward recurrence
                            -rlist[li - 1] * phi[li - 1]) # backward recurrence
        
        # dphi[li] = dPhiByPhi_recurrence(nu, li, rlist[li], x) * phi[li]

    return phi#, dphi